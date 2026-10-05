"""Дополнительные заголовки каждого запроса к LLM.

Задача — шлюз со своим ключом, трассировкой или маршрутизацией:
утилита должна уметь отдать любой заголовок, не правя код. Заголовки
при этом мультисет: одно имя может уйти на провод несколько раз.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock

import pytest
import requests

from hh_applicant_tool.ai import ChatOpenAI, normalize_headers
from hh_applicant_tool.main import HHApplicantTool


class _Response:
    status_code = 200

    def __init__(self, content: str = "ok") -> None:
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


def _session(response: _Response | None = None) -> requests.Session:
    """Настоящая сессия с подменённой отправкой.

    Подготовку запроса (тело, Content-Type) делает сама requests, и
    мок вернул бы вместо тела заглушку, поэтому мокается только send.
    """
    session = requests.Session()
    session.send = MagicMock(return_value=response or _Response())
    return session


def _client(**kwargs) -> tuple[ChatOpenAI, MagicMock]:
    session = kwargs.pop("session", None) or _session()
    client = ChatOpenAI(
        api_key="key",
        base_url="https://example.test/v1/chat/completions",
        session=session,
        **kwargs,
    )
    return client, session.send


def _wire_values(send: MagicMock, name: str) -> list[str]:
    """Все значения заголовка с этим именем, в том числе повторяющиеся.

    Имена заголовков регистронезависимы, поэтому искать можно в любом
    написании.
    """
    prepared = send.call_args.args[0]
    return list(prepared.headers.getlist(name))


def _tool(**config) -> HHApplicantTool:
    tool = HHApplicantTool()
    tool.proxy_url = None
    tool.openai_timeout = None
    tool.openai_connect_timeout = None
    tool.openai_proxy_url = None
    tool.use_developer_role = None
    tool.__dict__["config"] = {
        "openai": {
            "api_key": "key",
            "base_url": "https://example.test/v1/chat/completions",
            **config.pop("openai", {}),
        },
        **config,
    }
    return tool


class TestNormalizeHeaders:
    def test_none_gives_empty(self):
        assert normalize_headers(None) == []

    def test_dict_becomes_pairs(self):
        assert normalize_headers({"X-Trace": "abc", "X-Env": "prod"}) == [
            ("X-Trace", "abc"),
            ("X-Env", "prod"),
        ]

    def test_values_become_strings(self):
        """Из JSON значения приходят числами, а заголовок должен быть
        строкой."""
        assert normalize_headers({"X-N": 5}) == [("X-N", "5")]

    def test_empty_name_and_value_skipped(self):
        assert normalize_headers(
            {"  ": "x", "X-Ok": None, "X-Keep": "y"}
        ) == [("X-Keep", "y")]

    def test_name_trimmed(self):
        assert normalize_headers({" X-Key ": "v"}) == [("X-Key", "v")]

    def test_not_a_dict_is_config_error(self):
        with pytest.raises(ValueError, match="extra_headers"):
            normalize_headers("X-Key: v")

    def test_list_of_pairs_supported(self):
        assert normalize_headers([["X-K", "a"], ["X-K", "b"]]) == [
            ("X-K", "a"),
            ("X-K", "b"),
        ]

    def test_broken_pair_is_config_error(self):
        with pytest.raises(ValueError, match="extra_headers"):
            normalize_headers([["X-K"]])

    def test_repeated_name_kept(self):
        """Словарь затирает повтор, а мультисет его хранит."""
        assert normalize_headers({"X-K": ["a", "b"]}) == [
            ("X-K", "a"),
            ("X-K", "b"),
        ]


class TestHeadersInRequest:
    def test_extra_headers_sent(self):
        client, send = _client(
            extra_headers={"X-Trace": "abc", "X-Env": "prod"}
        )

        client.complete("hi")

        assert _wire_values(send, "X-Trace") == ["abc"]
        assert _wire_values(send, "X-Env") == ["prod"]

    def test_authorization_sent(self):
        client, send = _client(extra_headers={"X-Trace": "abc"})

        client.complete("hi")

        assert _wire_values(send, "Authorization") == ["Bearer key"]

    def test_authorization_not_overridden_but_joined(self):
        """Свой Authorization не выпиливается: до шлюза доходят оба, какой
        считать своим — решает он."""
        client, send = _client(
            extra_headers={"Authorization": "Token secret"}
        )

        client.complete("hi")

        assert _wire_values(send, "Authorization") == [
            "Bearer key",
            "Token secret",
        ]

    def test_repeated_name_goes_twice(self):
        client, send = _client(extra_headers={"X-K": ["a", "b"]})

        client.complete("hi")

        assert _wire_values(send, "X-K") == ["a", "b"]

    def test_headers_reach_captcha_request(self):
        """Голосование по капче ходит через отдельные сессии на потоки —
        заголовки должны попадать и туда."""
        captcha_reply = _Response(
            '{"first_word": "alpha", "second_word": "beta"}'
        )
        client, _ = _client(
            session=_session(captcha_reply),
            extra_headers={"X-Trace": "abc"},
        )
        thread_session = _session(captcha_reply)
        client._tls.session = thread_session

        assert (
            client._read_captcha_once(
                b"image-bytes", 0.7, script="latin", throttle=False
            )
            == "alpha beta"
        )
        assert _wire_values(thread_session.send, "X-Trace") == ["abc"]
        assert _wire_values(thread_session.send, "Authorization") == [
            "Bearer key"
        ]

    def test_no_extra_headers_by_default(self):
        client, send = _client()

        client.complete("hi")

        assert _wire_values(send, "Authorization") == ["Bearer key"]
        assert client.extra_headers == []

    def test_bad_headers_rejected_on_creation(self):
        with pytest.raises(ValueError, match="extra_headers"):
            ChatOpenAI(
                api_key="key",
                base_url="https://example.test",
                extra_headers="X-Key: v",
            )


class TestHeadersOnTheWire:
    """Проверка на настоящем сокете: мультисетность requests теряет по
    дороге, и мок этого не покажет."""

    @staticmethod
    def _serve(seen: list[tuple[list[tuple[str, str]], bytes]]) -> str:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - имя из BaseHTTPRequestHandler
                body = self.rfile.read(
                    int(self.headers.get("Content-Length", 0) or 0)
                )
                seen.append((list(self.headers.items()), body))
                payload = b'{"choices": [{"message": {"content": "ok"}}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: object) -> None:
                return None

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"

    def test_two_authorization_arrive_separately(self):
        seen: list[tuple[list[tuple[str, str]], bytes]] = []
        client = ChatOpenAI(
            api_key="key",
            base_url=self._serve(seen),
            extra_headers={"Authorization": "Token second"},
            max_retries=0,
        )

        client.complete("hi")

        headers, _ = seen[0]
        assert [v for k, v in headers if k == "Authorization"] == [
            "Bearer key",
            "Token second",
        ]

    def test_repeated_name_arrives_twice(self):
        seen: list[tuple[list[tuple[str, str]], bytes]] = []
        client = ChatOpenAI(
            api_key="key",
            base_url=self._serve(seen),
            extra_headers=[["X-Trace", "one"], ["X-Trace", "two"]],
            max_retries=0,
        )

        client.complete("hi")

        headers, _ = seen[0]
        assert [v for k, v in headers if k == "X-Trace"] == ["one", "two"]

    def test_body_and_content_type_survive(self):
        """Ручная подготовка запроса не должна ломать ни тело, ни
        Content-Type, который ставит сам requests."""
        seen: list[tuple[list[tuple[str, str]], bytes]] = []
        client = ChatOpenAI(
            api_key="key",
            base_url=self._serve(seen),
            max_retries=0,
        )

        assert client.complete("hi") == "ok"

        headers, body = seen[0]
        assert json.loads(body) == {
            "model": None,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.0,
            "max_completion_tokens": 1000,
            "stream": False,
        }
        assert dict(headers)["Content-Type"] == "application/json"


class TestConfigWiring:
    def test_headers_from_openai_section(self):
        tool = _tool(openai={"extra_headers": {"X-Trace": "abc"}})

        client = tool.get_cover_letter_ai("prompt")

        assert client.extra_headers == [("X-Trace", "abc")]

    def test_absent_headers_give_empty(self):
        client = _tool().get_cover_letter_ai("prompt")

        assert client.extra_headers == []

    def test_headers_reach_every_purpose(self):
        tool = _tool(openai={"extra_headers": {"X-Trace": "abc"}})

        assert tool.get_captcha_ai().extra_headers == [("X-Trace", "abc")]
        assert tool.get_chat_ai("p").extra_headers == [("X-Trace", "abc")]
        assert tool.get_vacancy_filter_ai("p").extra_headers == [
            ("X-Trace", "abc")
        ]

    def test_purpose_adds_to_common(self):
        """Заголовки секции цели дописываются к общим, а не заменяют их:
        иначе openai_captcha молча убирала бы ключ шлюза."""
        tool = _tool(
            openai={"extra_headers": {"X-Trace": "abc", "X-Env": "prod"}},
            openai_captcha={"extra_headers": {"X-Env": "test"}},
        )

        client = tool.get_captcha_ai()

        assert client.extra_headers == [
            ("X-Trace", "abc"),
            ("X-Env", "prod"),
            ("X-Env", "test"),
        ]

    def test_purpose_headers_do_not_leak_to_others(self):
        tool = _tool(
            openai={"extra_headers": {"X-Trace": "abc"}},
            openai_cover_letter={"extra_headers": {"X-Letter": "1"}},
        )

        assert tool.get_cover_letter_ai("p").extra_headers == [
            ("X-Trace", "abc"),
            ("X-Letter", "1"),
        ]
        assert tool.get_chat_ai("p").extra_headers == [("X-Trace", "abc")]

    def test_purpose_only_headers_without_common(self):
        tool = _tool(openai_chat={"extra_headers": {"X-Chat": "1"}})

        assert tool.get_chat_ai("p").extra_headers == [("X-Chat", "1")]

    def test_bad_headers_in_config_reported(self):
        tool = _tool(openai={"extra_headers": ["X-Trace"]})

        with pytest.raises(ValueError, match="extra_headers"):
            tool.get_cover_letter_ai("prompt")