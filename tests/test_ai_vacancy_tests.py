"""Решение теста вакансии: отдельный клиент и отказ вместо ответа вслепую.

Зачем: тестовые вопросы — не сопроводительное письмо. Раньше они уходили
тем же клиентом, и модель, решая тест, получала промпт письма. Второе
важнее: если модель не выбрала вариант, подставлять первый вариант из
списка нельзя, это заведомо неверный ответ от имени соискателя.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hh_applicant_tool.operations.apply_vacancies import (
    TEST_CHOICE_PROMPT,
    Operation,
    UnansweredTest,
)

RESUME = {
    "id": "resume-1",
    "title": "Python разработчик",
    "alternate_url": "https://hh.ru/resume/resume-1",
}

VACANCY = {
    "id": "111",
    "name": "Senior Python",
    "alternate_url": "https://hh.ru/vacancy/111",
    "employer": {"id": "222", "name": "ООО Ромашка"},
}

FULL_RESUME = {
    "title": "Python разработчик",
    "skills": "Пишу на Python 12 лет",
    "skill_set": ["Python", "Django"],
    "experience": [
        {
            "company": "Ромашка",
            "position": "Backend-разработчик",
            "start": "2010-01-01",
            "end": "2022-01-01",
            "description": "Делал сервисы на Python",
        }
    ],
}

TESTS_DATA = {
    "111": {
        "uidPk": "uid",
        "guid": "guid",
        "startTime": "start",
        "required": "true",
        "tasks": [
            {
                "id": 1,
                "description": "Готовы ли вы переехать?",
                "candidateSolutions": [
                    {"id": "10", "text": "да"},
                    {"id": "11", "text": "нет"},
                ],
            }
        ],
    }
}


def make_operation(answer: str | None = "10") -> Operation:
    operation = Operation()
    operation._args = SimpleNamespace()
    operation.tool = MagicMock()
    operation.tool.xsrf_token = "xsrf"
    operation._get_vacancy_tests = lambda response_url: TESTS_DATA
    operation.cover_letter_ai = MagicMock()
    operation.test_ai = MagicMock()
    operation.test_ai.answer_test_question.return_value = answer
    operation.tool.session.post.return_value = MagicMock(
        json=lambda: {"success": "true"}
    )
    # Резюме и вакансию тянем текстом, как в боевом запуске: без этих
    # заглушек _analyze_resume_heavy уходит в сеть через api_client,
    # которого у голого Operation нет.
    operation._resume_analysis_cache = {}
    operation._get_full_resume = lambda resume_id: FULL_RESUME
    operation._get_full_vacancy = lambda vacancy: None
    return operation


class TestChoiceQuestion:
    def test_picks_variant_by_id(self):
        operation = make_operation("10")

        operation._solve_vacancy_test(
            vacancy_id="111",
            resume_hash="resume-1",
            vacancy=VACANCY,
            resume=RESUME,
        )

        payload = operation.tool.session.post.call_args.kwargs["data"]
        assert payload["task_1"] == "10"

    def test_does_not_use_letter_client(self):
        """Промпт письма в тесте сбивает модель, клиент должен быть свой."""
        operation = make_operation("10")

        operation._solve_vacancy_test(
            vacancy_id="111",
            resume_hash="resume-1",
            vacancy=VACANCY,
            resume=RESUME,
        )

        assert operation.test_ai.answer_test_question.call_count == 1
        operation.cover_letter_ai.complete.assert_not_called()

    def test_ids_are_passed_to_the_model(self):
        operation = make_operation("10")

        operation._solve_vacancy_test(
            vacancy_id="111",
            resume_hash="resume-1",
            vacancy=VACANCY,
            resume=RESUME,
        )

        prompt = operation.test_ai.answer_test_question.call_args.args[0]
        assert "https://hh.ru/vacancy/111" in prompt
        assert "резюме: resume-1" in prompt

    def test_resume_text_is_passed_to_the_model(self):
        """По одному id резюме модель о кандидате ничего не знает."""
        operation = make_operation("10")

        operation._solve_vacancy_test(
            vacancy_id="111",
            resume_hash="resume-1",
            vacancy=VACANCY,
            resume=RESUME,
        )

        prompt = operation.test_ai.answer_test_question.call_args.args[0]
        assert "[РЕЗЮМЕ]" in prompt
        assert "Пишу на Python 12 лет" in prompt
        assert "Senior Python" in prompt

    def test_empty_answer_skips_vacancy(self):
        """Неуверенность — повод не отвечать, а не гадать."""
        operation = make_operation(None)

        with pytest.raises(UnansweredTest):
            operation._solve_vacancy_test(
                vacancy_id="111",
                resume_hash="resume-1",
                vacancy=VACANCY,
                resume=RESUME,
            )

        operation.tool.session.post.assert_not_called()

    def test_no_variant_id_skips_vacancy(self):
        """Первый вариант из списка — это заведомо неверный ответ."""
        operation = make_operation(None)

        with pytest.raises(UnansweredTest):
            operation._solve_vacancy_test(
                vacancy_id="111",
                resume_hash="resume-1",
                vacancy=VACANCY,
                resume=RESUME,
            )

        operation.tool.session.post.assert_not_called()

    def test_prompt_allows_empty_answer(self):
        operation = make_operation("10")

        operation._solve_vacancy_test(
            vacancy_id="111",
            resume_hash="resume-1",
            vacancy=VACANCY,
            resume=RESUME,
        )

        prompt = operation.test_ai.answer_test_question.call_args.args[0]
        assert TEST_CHOICE_PROMPT in prompt
        assert "answer" in prompt


@pytest.fixture
def with_tasks():
    """Подменяет задачи теста на время теста."""

    def _with_tasks(tasks: list[dict]) -> None:
        TESTS_DATA["111"]["tasks"] = tasks

    yield _with_tasks

    TESTS_DATA["111"]["tasks"] = [
        {
            "id": 1,
            "description": "Готовы ли вы переехать?",
            "candidateSolutions": [
                {"id": "10", "text": "да"},
                {"id": "11", "text": "нет"},
            ],
        }
    ]


class TestTextQuestion:
    def test_text_answer_is_sent(self, with_tasks):
        with_tasks([{"id": 2, "description": "Расскажите о себе"}])
        operation = make_operation("Двенадцать лет в Python")

        operation._solve_vacancy_test(
            vacancy_id="111",
            resume_hash="resume-1",
            vacancy=VACANCY,
            resume=RESUME,
        )

        payload = operation.tool.session.post.call_args.kwargs["data"]
        assert payload["task_2_text"] == "Двенадцать лет в Python"

    def test_empty_text_answer_skips_vacancy(self, with_tasks):
        with_tasks([{"id": 2, "description": "Расскажите о себе"}])
        operation = make_operation(None)

        with pytest.raises(UnansweredTest):
            operation._solve_vacancy_test(
                vacancy_id="111",
                resume_hash="resume-1",
                vacancy=VACANCY,
                resume=RESUME,
            )

        operation.tool.session.post.assert_not_called()

    def test_link_question_stays_handwritten(self, with_tasks):
        """Вопрос со ссылкой ответа не ждёт: модель тут не нужна."""
        with_tasks(
            [{"id": 3, "description": "Гуглдок: https://docs.google.com/1"}]
        )
        operation = make_operation("10")

        operation._solve_vacancy_test(
            vacancy_id="111",
            resume_hash="resume-1",
            vacancy=VACANCY,
            resume=RESUME,
        )

        payload = operation.tool.session.post.call_args.kwargs["data"]
        assert "task_3_text" in payload
        operation.test_ai.complete.assert_not_called()


class TestUnansweredTestIsValueError:
    def test_callers_keeping_value_error_still_catch_it(self):
        """Код вокруг решения теста ловит ValueError."""
        assert issubclass(UnansweredTest, ValueError)


