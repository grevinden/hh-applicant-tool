"""Роль системного промпта: system или developer.

Зачем: если запрос идёт через шлюз с агентом, тот уже добавил свой
системный промпт, и второй `system` отправить нельзя — шлюз или модель
такой запрос отклоняют. Роль `developer` переопределяет основной
системный промпт, поэтому с ней схема работает через любой шлюз.
По умолчанию оставлен `system`: не все модели и локальные серверы
знают про `developer`.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest
import requests

from hh_applicant_tool.ai.openai import (
    DEFAULT_SYSTEM_ROLE,
    NO_SYSTEM_ROLE,
    SYSTEM_ROLES,
    CAPTCHA_SCRIPT_LATIN,
    ChatOpenAI,
    resolve_system_role,
)
from hh_applicant_tool.main import HHApplicantTool


class _Response:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "true"}}]}


def _client(**kwargs) -> tuple[ChatOpenAI, MagicMock]:
    # Сессия настоящая: подготовку запроса делает она сама, и тело
    # собирается в байты, а мок вернул бы вместо тела заглушку
    session = requests.Session()
    send = MagicMock(return_value=_Response())
    session.send = send
    client = ChatOpenAI(
        api_key="test-key",
        base_url="https://example.test/v1/chat/completions",
        model="test-model",
        system_prompt="Only accept Python roles",
        rate_limit=0,
        session=session,
        **kwargs,
    )
    return client, send


def _sent_messages(send: MagicMock) -> list[dict]:
    """Сообщения из тела отправленного запроса."""
    prepared = send.call_args.args[0]
    return json.loads(prepared.body)["messages"]


class TestSystemRoleInPayload:
    def test_default_role_is_system(self):
        assert DEFAULT_SYSTEM_ROLE == "system"

    def test_system_role_is_default(self):
        client, send = _client()

        client.complete("Вакансия: Python developer")

        messages = _sent_messages(send)
        assert messages[0]["role"] == "system"

    def test_developer_role_used_when_asked(self):
        client, send = _client(system_role="developer")

        client.complete("Вакансия: Python developer")

        messages = _sent_messages(send)
        assert messages[0]["role"] == "developer"
        assert messages[0]["content"] == "Only accept Python roles"

    def test_user_message_stays_user(self):
        """Меняется только системный промпт, запрос остаётся запросом."""
        client, send = _client(system_role="developer")

        client.complete("Вакансия: Python developer")

        messages = _sent_messages(send)
        assert messages[1] == {
            "role": "user",
            "content": "Вакансия: Python developer",
        }

    def test_known_roles(self):
        assert SYSTEM_ROLES == ("system", "developer")


class TestResolveSystemRole:
    """Роль приходит из конфига, поэтому лишнее значение не должно
    ронять прогон."""

    def test_absent_value_defaults_to_system(self):
        assert resolve_system_role(None) == "system"

    def test_system_and_developer_kept(self):
        assert resolve_system_role("system") == "system"
        assert resolve_system_role("developer") == "developer"

    def test_case_and_spaces_ignored(self):
        assert resolve_system_role(" Developer ") == "developer"

    def test_user_means_no_system_message(self):
        assert resolve_system_role(NO_SYSTEM_ROLE) is None

    def test_unknown_value_means_no_system_message(self):
        """Опечатка не должна ронять прогон: запрос уходит без
        системного сообщения, о значении пишется в лог."""
        assert resolve_system_role("sistem") is None

    def test_unknown_value_warned(self, caplog):
        with caplog.at_level(logging.WARNING):
            resolve_system_role("sistem")

        assert "sistem" in caplog.text

    def test_known_value_not_warned(self, caplog):
        with caplog.at_level(logging.WARNING):
            resolve_system_role("developer")

        assert caplog.text == ""


class TestCaptchaPayloadRole:
    def test_developer_role_in_captcha_payload(self):
        """Капча идёт через тот же клиент, роль должна быть и там."""
        client = ChatOpenAI.__new__(ChatOpenAI)
        client.system_role = "developer"

        payload = client._captcha_payload(
            "YmFzZTY0", "image/png", 0.7, CAPTCHA_SCRIPT_LATIN
        )

        assert payload["messages"][0]["role"] == "developer"

    def test_default_role_in_captcha_payload(self):
        client = ChatOpenAI.__new__(ChatOpenAI)

        payload = client._captcha_payload(
            "YmFzZTY0", "image/png", 0.7, CAPTCHA_SCRIPT_LATIN
        )

        assert payload["messages"][0]["role"] == "system"


def _tool(**config) -> HHApplicantTool:
    tool = HHApplicantTool()
    # Атрибуты, которые обычно ставит _assign_args из аргументов командной
    # строки: тут клиент собирается напрямую из конфига
    tool.proxy_url = None
    tool.openai_timeout = None
    tool.openai_connect_timeout = None
    tool.openai_proxy_url = None
    tool.__dict__["config"] = {
        "openai": {
            "api_key": "key",
            "base_url": "https://example.test/v1/chat/completions",
            **config,
        }
    }
    return tool


class TestToolWiring:
    def test_default_role_from_tool(self):
        client = _tool().get_vacancy_filter_ai("prompt")
        assert client.system_role == "system"

    def test_config_system_role(self):
        client = _tool(system_role="system").get_vacancy_filter_ai("prompt")
        assert client.system_role == "system"

    def test_config_developer_role(self):
        client = _tool(system_role="developer").get_vacancy_filter_ai("prompt")
        assert client.system_role == "developer"

    def test_config_role_without_system_message(self):
        """Всё, что не system и не developer, — запрос без системного
        сообщения."""
        client = _tool(system_role="user").get_vacancy_filter_ai("prompt")
        assert client.system_role is None

    def test_role_reaches_every_purpose(self):
        """Капча, чат, письмо и фильтр — один и тот же клиент."""
        tool = _tool(system_role="developer")

        assert tool.get_captcha_ai().system_role == "developer"
        assert tool.get_chat_ai("prompt").system_role == "developer"
        assert tool.get_cover_letter_ai("prompt").system_role == "developer"
        assert tool.get_vacancy_filter_ai("prompt").system_role == "developer"

    def test_no_role_flag_in_parser(self):
        """Роль задаёт модель в конфиге, а не флаг запуска."""
        parser = HHApplicantTool()._parser

        with pytest.raises(SystemExit):
            parser.parse_args(["apply", "--use-developer-role"])


class TestNoSystemMessage:
    def test_system_prompt_not_sent(self):
        client, send = _client(system_role="user")

        client.complete("Вакансия: Python developer")

        messages = _sent_messages(send)
        assert [m["role"] for m in messages] == ["user"]

    def test_system_prompt_moves_to_user_message(self):
        """Промпт без системного сообщения терять нельзя: он уезжает в
        пользовательское."""
        client, send = _client(system_role="user")

        client.complete("hi")

        messages = _sent_messages(send)
        assert len(messages) == 1
        assert messages[0] == {
            "role": "user",
            "content": "Only accept Python roles\n\nhi",
        }

    def test_message_survives_without_system_prompt(self):
        """Без промпта склеивать нечего: сообщение уходит как есть."""
        session = requests.Session()
        send = MagicMock(return_value=_Response())
        session.send = send
        client = ChatOpenAI(
            api_key="test-key",
            base_url="https://example.test/v1/chat/completions",
            model="test-model",
            system_prompt=None,
            system_role="user",
            rate_limit=0,
            session=session,
        )

        client.complete("hi")

        assert _sent_messages(send) == [{"role": "user", "content": "hi"}]

    def test_captcha_payload_has_no_system_message(self):
        client = ChatOpenAI.__new__(ChatOpenAI)
        client.system_role = None

        payload = client._captcha_payload(
            "YmFzZTY0", "image/png", 0.7, CAPTCHA_SCRIPT_LATIN
        )

        assert [m["role"] for m in payload["messages"]] == ["user"]

    def test_captcha_keeps_rules_in_user_text(self):
        client = ChatOpenAI.__new__(ChatOpenAI)
        client.system_role = None

        payload = client._captcha_payload(
            "YmFzZTY0", "image/png", 0.7, CAPTCHA_SCRIPT_LATIN
        )

        text = payload["messages"][0]["content"][1]["text"]
        assert client.CAPTCHA_PROMPT_COMMON in text
        assert client.CAPTCHA_USER_PROMPT[CAPTCHA_SCRIPT_LATIN] in text