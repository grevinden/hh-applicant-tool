"""Предпросмотр (--dry-run) тоже уважает --max-responses.

Иначе dry-run проходил по всей выдаче: счётчик откликов не рос, лимит
никогда не срабатывал, и предпросмотр генерировал письма для всех
вакансий подряд.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from hh_applicant_tool.operations.apply_vacancies import Operation

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


def make_operation(max_responses: int) -> Operation:
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
    operation.max_responses = max_responses
    operation.dry_run = True
    operation.excluded_filter = None
    operation.message_prompt = "Напиши письмо"
    operation.force_message = True
    operation.cover_letter = "письмо"
    operation.cover_letter_ai = MagicMock()
    operation.cover_letter_ai.complete.return_value = "Здравствуйте!"
    operation.editor_ai = None
    operation._resume_analysis_cache = {}
    operation.__dict__["api_client"] = MagicMock()
    operation.api_client.get.return_value = {
        "id": "111",
        "name": "Senior Python",
        "description": "<p>Описание вакансии</p>",
        "employer": {"id": "222", "name": "ООО Ромашка"},
    }
    second = dict(
        VACANCY, id="112", alternate_url="https://hh.ru/vacancy/112"
    )
    operation._get_vacancies = lambda resume_id=None, resume_title="": (
        iter([dict(VACANCY), second])
    )
    return operation


class TestDryRunMaxResponses:
    def test_dry_run_stops_after_max_responses(self):
        operation = make_operation(max_responses=1)

        operation._apply_resume(
            resume=RESUME,
            user={"first_name": "Антон"},
            seen_employers=set(),
        )

        assert operation.cover_letter_ai.complete.call_count == 1

    def test_dry_run_without_limit_letters_every_vacancy(self):
        operation = make_operation(max_responses=0)

        operation._apply_resume(
            resume=RESUME,
            user={"first_name": "Антон"},
            seen_employers=set(),
        )

        assert operation.cover_letter_ai.complete.call_count == 2
