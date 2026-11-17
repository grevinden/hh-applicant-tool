"""Вход на hh по логину и паролю.

Раньше авторизация умела только одноразовый код: hh не всегда его шлет,
а другого пути в программе не было. Теперь пароль можно передать вторым
позиционным аргументом, а капча на этом шаге решается тем же способом,
что и при запросе кода.

Сеть не используется: подменяем session.post и распознавание капчи.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from hh_applicant_tool.operations.authorize import Operation
from hh_applicant_tool.tool import OperationError

REDIRECT_URL = (
    "https://hh.ru/account/login?role=applicant&backurl=%2Foauth%2Fauthorize"
)
# Настоящий публичный ключ нужен, потому что loginTrustFlags шифруются им
PUBKEY = (
    rsa.generate_private_key(public_exponent=65537, key_size=2048)
    .public_key()
    .public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    .decode()
)


class FakeResponse:
    status_code = 200

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.request = MagicMock(method="POST")
        self.url = REDIRECT_URL

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def post(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        payload = self.payloads.pop(0) if self.payloads else {}
        return FakeResponse(payload)


def build_operation(payloads: list[dict]) -> tuple[Operation, FakeSession]:
    session = FakeSession(payloads)
    tool = MagicMock()
    tool.session = session
    tool.get_captcha_image.return_value = MagicMock(key="captcha-key")
    tool.captcha_ai.recognize_text.return_value = "abcde"

    op = Operation()
    op._tool = tool
    return op, session


def test_parser_accepts_optional_password() -> None:
    parser = argparse.ArgumentParser()
    Operation().setup_parser(parser)

    args = parser.parse_args(["+79250000000"])
    assert args.username == "+79250000000"
    assert args.password is None

    args = parser.parse_args(["+79250000000", "secret"])
    assert args.password == "secret"


def test_password_login_posts_credentials() -> None:
    op, session = build_operation([{}])

    op._login_by_password(
        "79250000000",
        "secret",
        redirect_url=REDIRECT_URL,
        xsrf="xsrf-token",
        pubkey=PUBKEY,
        login_headers={"X-Xsrftoken": "xsrf-token"},
    )

    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == REDIRECT_URL
    assert call["data"]["username"] == "79250000000"
    assert call["data"]["password"] == "secret"
    assert call["data"]["accountType"] == "APPLICANT"
    assert call["data"]["remember"] == "true"
    assert "loginTrustFlags" in call["data"]
    # Пароль уходит в теле, а не в query string
    assert "files" not in call


def test_password_login_reports_error() -> None:
    op, _ = build_operation([{"loginError": {"code": "wrongPassword"}}])

    with pytest.raises(OperationError, match="wrongPassword"):
        op._login_by_password(
            "79250000000",
            "bad",
            redirect_url=REDIRECT_URL,
            xsrf="xsrf-token",
            pubkey=PUBKEY,
            login_headers={},
        )


def test_password_login_solves_captcha_and_retries() -> None:
    op, session = build_operation(
        [{"hhcaptcha": {"captchaState": "state-1"}}, {}]
    )

    op._login_by_password(
        "79250000000",
        "secret",
        redirect_url=REDIRECT_URL,
        xsrf="xsrf-token",
        pubkey=PUBKEY,
        login_headers={},
    )

    assert len(session.calls) == 2
    retry = session.calls[1]["data"]
    assert retry["captchaText"] == "abcde"
    assert retry["captchaKey"] == "captcha-key"
    assert retry["captchaState"] == "state-1"
    # Логин и пароль не теряются при повторной отправке
    assert retry["password"] == "secret"
