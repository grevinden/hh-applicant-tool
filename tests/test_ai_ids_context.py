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
    MAX_ID_DEPTH,
    MAX_ID_LIST_ITEMS,
    MAX_IDS,
    Operation,
    collect_ids,
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


class TestCollectIds:
    def test_nested_id_keeps_path(self):
        found = collect_ids({"employer": {"id": "222"}})
        assert found == {"employer.id": "222"}

    def test_plain_id_without_prefix(self):
        assert collect_ids({"id": "1"}) == {"id": "1"}

    def test_suffixed_ids(self):
        found = collect_ids({"vacancy_id": "1", "resume_id": "r", "user_id": "u"})
        assert found == {
            "vacancy_id": "1",
            "resume_id": "r",
            "user_id": "u",
        }

    def test_list_items_are_indexed(self):
        found = collect_ids({"contacts": [{"id": "c1"}, {"id": "c2"}]})
        assert found == {"contacts[0].id": "c1", "contacts[1].id": "c2"}

    def test_long_list_is_trimmed(self):
        """Список бывает длинным, а в промпт всё это не нужно."""
        found = collect_ids({"items": [{"id": str(i)} for i in range(50)]})
        assert len(found) == MAX_ID_LIST_ITEMS

    def test_non_scalar_ids_skipped(self):
        """`id: true` в ответах hh не бывает, но мусор в промпте хуже."""
        found = collect_ids({"id": True, "employer_id": None})
        assert found == {}

    def test_depth_is_limited(self):
        data: dict = {}
        current = data
        for _ in range(MAX_ID_DEPTH + 3):
            current["nested"] = {}
            current = current["nested"]
        current["id"] = "deep"

        found = collect_ids(data)
        assert "id" not in found.values()

    def test_total_size_is_limited(self):
        """Много вложенных списков не должны раздувать запрос."""
        data = {f"key{i}": [{"id": f"{i}-{j}"} for j in range(10)] for i in range(20)}
        assert len(collect_ids(data)) <= MAX_IDS


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

        assert "[ИДЕНТИФИКАТОРЫ ДЛЯ ДОЗАГРУЗКИ]" in text
        assert "вакансия: 111 (https://hh.ru/vacancy/111)" in text
        assert "работодатель: 222 — ООО Ромашка" in text
        assert "резюме: resume-1 (https://hh.ru/resume/resume-1)" in text

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

        assert "работодатель: 999 — АО Ромашка" in text
        assert "222" not in text

    def test_other_ids_are_listed_once(self):
        full = dict(VACANCY, contacts=[{"id": "c1"}])

        text = self.operation()._build_ids_context(full_vacancy=full)

        assert "прочие id из ответа hh.ru: contacts[0].id=c1" in text
        # Вакансия и работодатель уже своими строками
        assert text.count("вакансия: 111") == 1

    def test_missing_employer_id_is_not_fatal(self):
        vacancy = {"id": "111", "employer": {"name": "Без id"}}

        text = self.operation()._build_ids_context(vacancy=vacancy)

        assert "вакансия: 111" in text
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
        assert "вакансия: 111" in prompt
        assert "работодатель: 222" in prompt
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
        assert "[ИДЕНТИФИКАТОРЫ ДЛЯ ДОЗАГРУЗКИ]" not in prompt


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
        assert "[ИДЕНТИФИКАТОРЫ ДЛЯ ДОЗАГРУЗКИ]" in msg
        assert "вакансия: 111" in msg
        assert "работодатель: 222" in msg
        assert "резюме: resume-1" in msg
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
        assert "[ИДЕНТИФИКАТОРЫ ДЛЯ ДОЗАГРУЗКИ]" in prompt
        assert "вакансия: 111" in prompt
        assert "резюме: resume-1" in prompt