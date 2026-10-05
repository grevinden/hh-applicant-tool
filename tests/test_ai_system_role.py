"""Роль системного промпта: system или developer.

Зачем: если запрос идёт через шлюз с агентом, тот уже добавил свой
системный промпт, и второй `system` отправить нельзя — шлюз или модель
такой запрос отклоняют. Роль `developer` переопределяет основной
системный промпт, поэтому с ней схема работает через любой шлюз.
По умолчанию оставлен `system`: не все модели и локальные серверы
знают про `developer`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hh_applicant_tool.ai.openai import (
    DEFAULT_SYSTEM_ROLE,
    SYSTEM_ROLES,
    CAPTCHA_SCRIPT_LATIN,
    ChatOpenAI,
)
from hh_applicant_tool.main import HHApplicantTool


class _Response:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "true"}}]}


def _client(**kwargs) -> tuple[ChatOpenAI, MagicMock]:
    session = MagicMock()
    session.post.return_value = _Response()
    client = ChatOpenAI(
        api_key="test-key",
        base_url="https://example.test/v1/chat/completions",
        model="test-model",
        system_prompt="Only accept Python roles",
        rate_limit=0,
        session=session,
        **kwargs,
    )
    return client, session


class TestSystemRoleInPayload:
    def test_default_role_is_system(self):
        assert DEFAULT_SYSTEM_ROLE == "system"

    def test_system_role_is_default(self):
        client, session = _client()

        client.complete("Вакансия: Python developer")

        messages = session.post.call_args.kwargs["json"]["messages"]
        assert messages[0]["role"] == "system"

    def test_developer_role_used_when_asked(self):
        client, session = _client(system_role="developer")

        client.complete("Вакансия: Python developer")

        messages = session.post.call_args.kwargs["json"]["messages"]
        assert messages[0]["role"] == "developer"
        assert messages[0]["content"] == "Only accept Python roles"

    def test_user_message_stays_user(self):
        """Меняется только системный промпт, запрос остаётся запросом."""
        client, session = _client(system_role="developer")

        client.complete("Вакансия: Python developer")

        messages = session.post.call_args.kwargs["json"]["messages"]
        assert messages[1] == {
            "role": "user",
            "content": "Вакансия: Python developer",
        }

    def test_unknown_role_rejected(self):
        with pytest.raises(ValueError, match="роль системного промпта"):
            _client(system_role="user")

    def test_known_roles(self):
        assert SYSTEM_ROLES == ("system", "developer")


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
    # Флаг не задан, как при запуске без --use-developer-role
    tool.use_developer_role = None
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

    def test_flag_switches_role(self):
        tool = _tool()
        tool.use_developer_role = True

        client = tool.get_vacancy_filter_ai("prompt")

        assert client.system_role == "developer"

    def test_config_switches_role(self):
        client = _tool(use_developer_role=True).get_vacancy_filter_ai("prompt")
        assert client.system_role == "developer"

    def test_flag_wins_over_config(self):
        """Флаг важнее config.json: так можно один раз переключиться на
        другой шлюз, не правя конфиг."""
        tool = _tool(use_developer_role=True)
        tool.use_developer_role = False

        client = tool.get_vacancy_filter_ai("prompt")

        assert client.system_role == "system"

    def test_role_reaches_every_purpose(self):
        """Капча, чат, письмо и фильтр — один и тот же клиент."""
        tool = _tool()
        tool.use_developer_role = True

        assert tool.get_captcha_ai().system_role == "developer"
        assert tool.get_chat_ai("prompt").system_role == "developer"
        assert tool.get_cover_letter_ai("prompt").system_role == "developer"
        assert tool.get_vacancy_filter_ai("prompt").system_role == "developer"


class TestFlag:
    def test_flag_parses(self):
        parser = HHApplicantTool()._parser
        args = parser.parse_args(["--use-developer-role", "apply"])
        assert args.use_developer_role is True

    def test_alias_parses(self):
        parser = HHApplicantTool()._parser
        args = parser.parse_args(["--developer-role", "apply"])
        assert args.use_developer_role is True

    def test_defaults_to_system(self):
        parser = HHApplicantTool()._parser
        args = parser.parse_args(["apply"])
        # Не задано, чтобы дать шанс config.json; system подставит
        # сам клиент
        assert args.use_developer_role is None
        tool = _tool()
        tool.use_developer_role = args.use_developer_role
        assert tool.get_vacancy_filter_ai("prompt").system_role == "system"