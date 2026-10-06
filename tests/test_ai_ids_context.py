"""Идентификаторы для дозагрузки инструментами шлюза.

Зачем это: через шлюз с MCP-инструментами модель может сама дотянуть
вакансию, работодателя и резюме, если знает их id. Без id анализ целиком
зависит от того, сколько данных успела прислать утилита, поэтому в
тяжёлом анализе и в сопроводительном письме отдаём всё, что нашли в
ответах hh.ru.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from hh_applicant_tool.operations.apply_vacancies import (
    AI_TOOLS_HINT,
    Operation,
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


class TestBuildIdsContext:
    def operation(self) -> Operation:
        operation = Operation()
        operation._args = SimpleNamespace()
        operation.tool = MagicMock()
        return operation

    def test_vacancy_employer_and_resume(self):
        text = self.operation()._build_ids_context(
            vacancy=VACANCY, resume=RESUME
        )

        assert "[ДАННЫЕ ДЛЯ ДОЗАГРУЗКИ]" in text
        assert "- вакансия: https://hh.ru/vacancy/111" in text
        assert "- работодатель: https://hh.ru/employer/222 — ООО Ромашка" in text
        assert "- резюме: resume-1" in text

    def test_resume_has_no_link(self):
        """Ссылка на резюме модели бесполезна — шлюз грузит по id."""
        text = self.operation()._build_ids_context(
            vacancy=VACANCY, resume=RESUME
        )

        assert "hh.ru/resume" not in text

    def test_no_other_ids(self):
        """Остальные идентификаторы ответа hh.ru только захламляли
        запрос и путали модель."""
        full = dict(VACANCY, contacts=[{"id": "c1"}], contacts_id="x")

        text = self.operation()._build_ids_context(full_vacancy=full)

        assert "c1" not in text
        assert "contacts" not in text

    def test_vacancy_and_employer_without_bare_ids(self):
        """Шлюз грузит по ссылке, голый номер рядом с ней лишний."""
        text = self.operation()._build_ids_context(
            vacancy=VACANCY, resume=RESUME
        )

        assert "- вакансия: 111 " not in text
        assert "- работодатель: 222" not in text

    def test_full_vacancy_wins_over_search_response(self):
        """Полный ответ вакансии богаче: id работодателя может не быть в
        поисковой выдаче."""
        full = {
            "id": "111",
            "employer": {"id": "999", "name": "АО Ромашка"},
            "contacts": [{"id": "c1"}],
        }

        text = self.operation()._build_ids_context(
            vacancy={"id": "111", "employer": {"id": "222"}},
            full_vacancy=full,
            resume=RESUME,
        )

        assert "https://hh.ru/employer/999 — АО Ромашка" in text

    def test_missing_employer_id_is_not_fatal(self):
        vacancy = {"id": "111", "employer": {"name": "Без id"}}

        text = self.operation()._build_ids_context(vacancy=vacancy)

        assert "https://hh.ru/vacancy/111" in text
        assert "работодатель" not in text

    def test_url_is_built_when_absent(self):
        text = self.operation()._build_ids_context(
            vacancy={"id": "111", "employer": {}}
        )
        assert "https://hh.ru/vacancy/111" in text

    def test_nothing_to_report(self):
        assert self.operation()._build_ids_context() == ""

    def test_resume_without_id(self):
        text = self.operation()._build_ids_context(
            vacancy=VACANCY, resume={"title": "без id"}
        )
        assert "резюме:" not in text

    def test_uncertain_answer_rule_present(self):
        """Неуверенность — повод промолчать, а не выдумать."""
        text = self.operation()._build_ids_context(
            vacancy=VACANCY, resume=RESUME
        )

        assert "пустой ответ" in text


class TestHeavyFilterPrompt:
    """Тяжёлый анализ: id в запросе + подсказка про инструменты."""

    def operation(self) -> Operation:
        operation = Operation()
        operation._args = SimpleNamespace()
        operation.tool = MagicMock()
        # api_client — cached_property, задаём через __dict__
        operation.__dict__["api_client"] = MagicMock()
        operation.api_client.get.return_value = VACANCY
        operation.vacancy_filter_ai = MagicMock()
        operation.vacancy_filter_ai.complete.return_value = '{"suitable": true}'
        return operation

    def test_ids_land_in_the_request(self):
        operation = self.operation()

        operation._is_vacancy_suitable_heavy(VACANCY, resume=RESUME)

        prompt = operation.vacancy_filter_ai.complete.call_args.args[0]
        assert "[ДАННЫЕ ДЛЯ ДОЗАГРУЗКИ]" in prompt
        assert "вакансия: https://hh.ru/vacancy/111" in prompt
        assert "работодатель: https://hh.ru/employer/222" in prompt
        assert "резюме: resume-1" in prompt

    def test_system_prompt_mentions_tools(self):
        operation = self.operation()

        prompt = operation._build_filter_system_prompt_heavy("resume")

        assert AI_TOOLS_HINT in prompt

    def test_light_filter_stays_cheap(self):
        """Лёгкий режим создан, чтобы не платить за лишний запрос и текст:
            идентификаторы и подсказка там не нужны."""
        operation = self.operation()

        prompt = operation._build_filter_system_prompt_light("resume")

        assert AI_TOOLS_HINT not in prompt

    def test_light_prompt_has_no_ids(self):
        operation = self.operation()

        operation._is_vacancy_suitable_light(VACANCY)

        prompt = operation.vacancy_filter_ai.complete.call_args.args[0]
        assert "[ДАННЫЕ ДЛЯ ДОЗАГРУЗКИ]" not in prompt


class TestCoverLetterPrompt:
    """Та же схема в сопроводительном письме."""

    def test_letter_request_carries_ids(self):
        operation = Operation()
        operation._args = SimpleNamespace(
            ai_rate_limit=0,
            send_email=False,
            skip_tests=False,
        )
        operation.tool = MagicMock()
        operation.tool.storage = MagicMock()
        operation.ai_filter = None
        operation.ai_filter_prompt = None
        operation.vacancy_filter_ai = None
        operation.max_responses = 0
        operation.dry_run = True
        operation.excluded_filter = None
        operation.message_prompt = "Напиши письмо"
        operation.force_message = True
        operation.cover_letter = "письмо"
        operation.cover_letter_ai = MagicMock()
        operation.cover_letter_ai.complete.return_value = "Здравствуйте!"
        operation._get_vacancies = lambda resume_id=None, resume_title="": (
            iter([dict(VACANCY)])
        )

        operation._apply_resume(
            resume=RESUME,
            user={"first_name": "Антон"},
            seen_employers=set(),
        )

        msg = operation.cover_letter_ai.complete.call_args.args[0]
        assert "[ДАННЫЕ ДЛЯ ДОЗАГРУЗКИ]" in msg
        assert "вакансия: https://hh.ru/vacancy/111" in msg
        assert "работодатель: https://hh.ru/employer/222" in msg
        assert "резюме: resume-1" in msg
        assert "hh.ru/resume" not in msg
        assert AI_TOOLS_HINT in msg


class TestIdsInFlow:
    """Сквозная проверка: id доезжают до реального запроса к модели."""

    def test_heavy_run_sends_ids_to_ai(self):
        operation = Operation()
        operation._args = SimpleNamespace(
            ai_rate_limit=0,
            send_email=False,
            skip_tests=False,
        )
        operation.tool = MagicMock()
        operation.tool.storage = MagicMock()
        filter_ai = MagicMock()
        filter_ai.complete.return_value = '{"suitable": true}'
        operation.tool.get_vacancy_filter_ai.return_value = filter_ai
        operation.ai_filter = "heavy"
        operation.ai_filter_prompt = None
        operation.max_responses = 0
        operation.dry_run = True
        operation.excluded_filter = None
        operation.message_prompt = "Напиши письмо"
        operation.force_message = False
        operation.cover_letter = "письмо"
        operation.cover_letter_ai = None
        operation._analyze_resume_heavy = lambda resume: "resume details"
        operation._resume_analysis_cache = {}
        operation.__dict__["api_client"] = MagicMock()
        operation.api_client.get.return_value = VACANCY
        operation._get_vacancies = lambda resume_id=None, resume_title="": (
            iter([dict(VACANCY)])
        )

        operation._apply_resume(
            resume=RESUME,
            user={"first_name": "Антон"},
            seen_employers=set(),
        )

        system_prompt = operation.tool.get_vacancy_filter_ai.call_args.args[0]
        assert "resume details" in system_prompt
        assert AI_TOOLS_HINT in system_prompt

        prompt = filter_ai.complete.call_args.args[0]
        assert "[ДАННЫЕ ДЛЯ ДОЗАГРУЗКИ]" in prompt
        assert "вакансия: https://hh.ru/vacancy/111" in prompt
        assert "резюме: resume-1" in prompt