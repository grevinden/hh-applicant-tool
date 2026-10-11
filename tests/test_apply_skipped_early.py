"""Проверка «вакансия уже отклонена» идёт до цикла вызовов модели.

Раньше эта проверка жила внутри блока ``--ai-filter``, поэтому без
фильтра (например, при одном ``--use-ai``) программа заново писала
письмо и решала тест для вакансии, отброшенной в прошлом запуске.
Теперь вакансия отсеивается в начале обработки, до обращений к модели.
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


def make_operation() -> Operation:
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


class TestAlreadySkippedIsCheckedBeforeModel:
    def test_skipped_vacancy_is_dropped_before_any_model_call(self):
        operation = make_operation()
        events: list[tuple[str, str]] = []

        def is_skipped(vacancy, resume_id=None):
            events.append(("skip", vacancy["id"]))
            return vacancy["id"] == "111"

        def complete(msg):
            events.append(("letter", ""))
            return "Здравствуйте!"

        operation._is_vacancy_already_skipped = is_skipped
        operation.cover_letter_ai.complete.side_effect = complete

        operation._apply_resume(
            resume=RESUME,
            user={"first_name": "Антон"},
            seen_employers=set(),
        )

        # Отклонённую вакансию проверяем раньше письма. Письмо пишем
        # только для второй вакансии, то есть без AI-фильтра тоже.
        assert events == [("skip", "111"), ("skip", "112"), ("letter", "")]
