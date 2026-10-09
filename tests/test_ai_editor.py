"""Второй вызов модели — редактор готового сообщения.

Письмо и сообщение в чате иногда уходят исковерканными: с иероглифами,
случайными английскими словами и машинным тоном. Поэтому после генерации
текст прогоняется через отдельного клиента-редактора, который получает
только сам текст (без данных о вакансии) и очеловечивает его.

Редактор не должен срывать отправку: если он упал, уходит исходный текст.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from hh_applicant_tool.ai.base import AIError
from hh_applicant_tool.operations.apply_vacancies import Operation as Apply
from hh_applicant_tool.operations.reply_employers import (
    Operation as Reply,
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

NEGOTIATION = {
    "id": "neg1",
    "updated_at": "2024-01-01T00:00:00+03:00",
    "state": {"id": "response"},
    "resume": {"id": "resume-1"},
    "vacancy": {
        "name": "Python разработчик",
        "alternate_url": "https://hh.ru/vacancy/111",
        "employer": {"id": "222", "name": "ООО Ромашка"},
    },
}

MESSAGES = {
    "items": [
        {
            "text": "Здравствуйте, вы нам подходите",
            "created_at": "2024-01-01T00:00:00+03:00",
            "author": {"participant_type": "employer"},
        },
        {
            "text": "Спасибо",
            "created_at": "2024-01-01T00:05:00+03:00",
            "author": {"participant_type": "applicant"},
        },
    ],
    "pages": 1,
}


class TestApplyEditor:
    def _operation(self) -> Apply:
        operation = Apply()
        operation._args = SimpleNamespace(
            ai_rate_limit=0,
            send_email=False,
            skip_tests=True,
        )
        operation.tool = MagicMock()
        operation.tool.storage = MagicMock()
        operation.ai_filter = None
        operation.ai_filter_prompt = None
        operation.vacancy_filter_ai = None
        operation.max_responses = 0
        operation.dry_run = False
        operation.excluded_filter = None
        operation._resume_analysis_cache = {}
        operation.message_prompt = "Напиши письмо"
        operation.force_message = True
        operation.cover_letter = "письмо"
        operation.cover_letter_ai = MagicMock()
        operation.cover_letter_ai.complete.return_value = "Здравствуйте!"
        operation.editor_ai = MagicMock()
        operation.editor_ai.complete.return_value = "Привет, я Антон!"
        operation._get_vacancies = lambda resume_id=None, resume_title="": (
            iter([dict(VACANCY)])
        )
        operation.__dict__["api_client"] = MagicMock()
        operation.api_client.get.side_effect = lambda url: (
            {"description": "<p>Описание вакансии</p>"}
            if "vacancies" in url
            else {"title": "Python разработчик", "skills": "О себе"}
        )
        operation.api_client.post.return_value = {}
        return operation

    def test_editor_receives_only_the_letter(self):
        operation = self._operation()

        operation._apply_resume(
            resume=RESUME,
            user={"first_name": "Антон"},
            seen_employers=set(),
        )

        operation.editor_ai.complete.assert_called_once_with("Здравствуйте!")

    def test_edited_letter_is_what_gets_sent(self):
        operation = self._operation()

        operation._apply_resume(
            resume=RESUME,
            user={"first_name": "Антон"},
            seen_employers=set(),
        )

        params = operation.api_client.post.call_args.args[1]
        assert params["message"] == "Привет, я Антон!"

    def test_editor_failure_does_not_block_sending(self):
        operation = self._operation()
        operation.editor_ai.complete.side_effect = AIError("boom")

        operation._apply_resume(
            resume=RESUME,
            user={"first_name": "Антон"},
            seen_employers=set(),
        )

        params = operation.api_client.post.call_args.args[1]
        assert params["message"] == "Здравствуйте!"


class TestReplyEditor:
    def _operation(self) -> Reply:
        operation = Reply()
        operation.tool = MagicMock()
        operation.tool.get_negotiations.return_value = [dict(NEGOTIATION)]
        operation.resume_id = None
        operation.period = None
        operation.only_invitations = False
        operation.dry_run = False
        operation.reply_message = "Отклик на %(vacancy_name)s"
        operation.message_prompt = "Ответь работодателю"
        operation.cover_letter_ai = None
        operation.editor_ai = MagicMock()
        operation.editor_ai.complete.return_value = "Отредактировано"
        operation.__dict__["api_client"] = MagicMock()
        operation.api_client.get.return_value = MESSAGES
        return operation

    def _run(self, operation: Reply) -> None:
        operation._reply_chats(
            user={
                "first_name": "Антон",
                "last_name": "",
                "email": "",
                "phone": "",
            },
            resumes=[{"id": "resume-1", "title": "Python разработчик"}],
            blacklist=set(),
        )

    def test_editor_receives_the_rendered_message(self):
        operation = self._operation()

        self._run(operation)

        operation.editor_ai.complete.assert_called_once_with(
            "Отклик на Python разработчик"
        )

    def test_edited_message_is_what_gets_sent(self):
        operation = self._operation()

        self._run(operation)

        assert (
            operation.api_client.post.call_args.kwargs["message"]
            == "Отредактировано"
        )

    def test_editor_failure_does_not_block_sending(self):
        operation = self._operation()
        operation.editor_ai.complete.side_effect = AIError("boom")

        self._run(operation)

        assert (
            operation.api_client.post.call_args.kwargs["message"]
            == "Отклик на Python разработчик"
        )
