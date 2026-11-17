from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urljoin, urlsplit

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from ..tool import BaseOperation, OperationError

if TYPE_CHECKING:
    from ..tool import HHApplicantTool


logger = logging.getLogger("hh-applicant-tool")

# Сколько раз пробуем пройти капчу на одном шаге авторизации
CAPTCHA_ATTEMPTS = 3


class Operation(BaseOperation):
    """Авторизация на сайте и через мобильное приложение"""

    __aliases__: list = ["authenticate", "auth", "login"]

    def setup_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("username", nargs="?", help="Email или телефон")
        parser.add_argument(
            "password",
            nargs="?",
            help=(
                "Пароль. Если не указан, код придет по SMS или на почту "
                "(hh не всегда отправляет SMS)"
            ),
        )

    def run(self, tool: HHApplicantTool, args) -> int | None:
        self._tool = tool
        self._args = args
        try:
            self._run()
        except KeyboardInterrupt:
            logger.warning("Операция прервана пользователем")
            return 1
        return 0

    def _run(self) -> None:
        tool = self._tool
        args = self._args
        api_client = tool.api_client
        storage = tool.storage
        session = tool.session

        username = (
            args.username
            or storage.settings.get_value("auth.username")
            or input("👤 Введите email или телефон: ")
        ).strip()
        if not username:
            raise OperationError("Empty username")
        logger.debug("authenticate with: %s", username)

        # Пароль необязателен: без него вход идет по одноразовому коду
        password = (args.password or "").strip() or None

        is_email = "@" in username
        if not is_email:
            username = self._normalize_phone_number(username)

        # https://api.hh.ru/openapi/redoc#section/Avtorizaciya/Zapros-avtorizacii-pod-drugim-polzovatelem
        authorize_url = api_client.oauth_client.authorize_url(
            force_login="true",
            role="applicant",
            force_role="true",
            skip_choose_account="true",
        )
        logger.info(
            f"Эту ссылку можно открыть в браузере для изучения процесса авторизации: {authorize_url=}"
        )
        # 1. Страница авторизации -> xsrf и публичный ключ
        r = session.get(authorize_url)
        # Если куки не протушли, то авторизация не пройдет

        redirect_url = r.url
        logger.debug(redirect_url)

        data = tool.parse_initial_state(r)
        xsrf = data["xsrfToken"]

        try:
            pubkey = data["loginTrustFlagsSecret"]
        except KeyError as ex:
            raise OperationError("Что-то пошло не так!") from ex

        # Эти заголовки у всех POST-запросов при авторизации
        # У них referer отличается только, но наврядли его гикто не проверяет
        login_headers = {
            "Accept": "application/json",
            "X-Hhtmfrom": "",
            "X-Hhtmsource": "account_login",
            "X-Requested-With": "XMLHttpRequest",
            "X-Xsrftoken": xsrf,
            "Referer": redirect_url,
        }

        if password:
            self._login_by_password(
                username,
                password,
                redirect_url=redirect_url,
                xsrf=xsrf,
                pubkey=pubkey,
                login_headers=login_headers,
            )
        else:
            self._request_otp(
                username,
                is_email,
                redirect_url=redirect_url,
                xsrf=xsrf,
                pubkey=pubkey,
                login_headers=login_headers,
            )
            self._confirm_otp(
                username,
                is_email,
                redirect_url=redirect_url,
                xsrf=xsrf,
                pubkey=pubkey,
                login_headers=login_headers,
            )

        # Теперь нужно проверить заголовки
        backurl = parse_qs(urlsplit(redirect_url).query)["backurl"][0]
        r = session.get(
            urljoin(redirect_url, backurl),
            headers={"Referer": redirect_url},
            allow_redirects=False,
        )

        logger.debug(f"{r.url} {r.status_code}")

        if r.status_code != 302 or not (location := r.headers.get("location")):
            raise OperationError(
                "Ответ не вляется редиректом или не содержит заголовка Location!"
            )

        sp = urlsplit(location)
        logger.debug(sp)

        if sp.scheme != "hhandroid" or sp.netloc != "oauthresponse":
            raise OperationError(
                f"Неправильный формат URI для получения кода авторизации: {location}"
            )

        auth_code = parse_qs(sp.query)["code"][0]

        # 5. Обмен кода на токен
        token = api_client.oauth_client.authenticate(auth_code)
        api_client.handle_access_token(token)
        storage.settings.set_value("auth.username", username)
        storage.settings.set_value("auth.last_login", datetime.now())
        # Куки уже лежат в tool.session — копировать ничего не нужно.
        print("🎉 Авторизация прошла успешно!")

    def _login_by_password(
        self,
        username: str,
        password: str,
        *,
        redirect_url: str,
        xsrf: str,
        pubkey: str,
        login_headers: dict[str, str],
    ) -> None:
        """Вход по логину и паролю.

        После успеха в сессии лежат авторизационные куки, а _run()
        завершает вход переходом на backurl и обменом кода на токен.
        """
        # Шлем на тот же URL страницы логина, в failUrl — он же без хоста
        sp = urlsplit(redirect_url)
        fail_url = f"{sp.path}?{sp.query}"

        payload = {
            "accountType": "APPLICANT",
            "failUrl": fail_url,
            "remember": "true",
            # Пароль подставлен менеджером паролей
            "loginTrustFlags": self._login_trust_flags(
                pubkey, {"password": {"suggest": True, "paste": False}}
            ),
            "username": username,
            "password": password,
            # Браузер всегда шлет это поле, при капче его перезапишет
            "captchaText": "",
        }

        for _ in range(CAPTCHA_ATTEMPTS):
            logger.debug(
                "password login params: %r", {**payload, "password": "***"}
            )
            res = self._post_json(redirect_url, payload, login_headers)

            if login_error := res.get("loginError"):
                code = login_error.get("code", "unknown")
                raise OperationError(f"Ошибка авторизации: {code}")

            captcha = res.get("hhcaptcha") or {}
            if "captchaState" not in captcha:
                logger.info("Вход по паролю выполнен")
                return

            if captcha.get("captchaError"):
                logger.warning("Капча была введена неверно!")

            payload |= self._captcha_fields(
                captcha, xsrf=xsrf, referer_url=redirect_url
            )

        raise OperationError("Не удалось пройти капчу при входе по паролю")

    def _request_otp(
        self,
        username: str,
        is_email: bool,
        *,
        redirect_url: str,
        xsrf: str,
        pubkey: str,
        login_headers: dict[str, str],
    ) -> None:
        """Просит hh выслать одноразовый код."""
        payload = {
            "login": username,
            "otpType": ["phone", "email"][is_email],
            "operationType": "applicant_otp_auth",
            "authScenario": "APPLICANT_AUTH",
            "formatPhone": "true",
            "isSignupPage": "false",
            "loginTrustFlags": self._login_trust_flags(pubkey),
            "captchaText": "",
        }

        otp_url = urljoin(redirect_url, "/account/otp_generate")

        for _ in range(CAPTCHA_ATTEMPTS):
            logger.debug("otp generate params: %r", payload)
            res = self._post_json(
                otp_url, payload, login_headers, multipart=True
            )

            if res.get("success"):
                logger.info("Код авторизации был выслан!")
                return

            # isBot еще интересное значение
            captcha = res.get("hhcaptcha") or {}
            if "captchaState" not in captcha:
                raise OperationError("Неправильный запрос!")

            if captcha.get("captchaError"):
                logger.warning("Капча была введена неверно!")

            payload |= self._captcha_fields(
                captcha, xsrf=xsrf, referer_url=redirect_url
            )

        raise OperationError("Не удалось пройти капчу при запросе кода")

    def _confirm_otp(
        self,
        username: str,
        is_email: bool,
        *,
        redirect_url: str,
        xsrf: str,
        pubkey: str,
        login_headers: dict[str, str],
    ) -> None:
        """Ввод кода из SMS или письма."""
        code_from = ["SMS", "Email"][is_email]
        code_icon = ["📱", "📧"][is_email]
        code = input(
            f"{code_icon} Введите полученный код из {code_from}: "
        ).strip()

        if not code:
            raise OperationError("Код подтверждения не может быть пустым.")

        payload = {
            "username": username,
            "code": code,
            "backurl": redirect_url,
            "operationType": "applicant_otp_auth",
            "isApplicantSignup": "false",
            "remember": "true",
            # Не знаю от чего их вычислять и нужны ли они тут
            "loginTrustFlags": self._login_trust_flags(pubkey),
            "authScenario": "APPLICANT_AUTH",
        }

        by_code_url = urljoin(redirect_url, "/account/login/by_code")

        for _ in range(CAPTCHA_ATTEMPTS):
            res = self._post_json(by_code_url, payload, login_headers)

            if res.get("success"):
                assert res.get("verification", {}).get("success")
                return

            captcha = res.get("hhcaptcha") or {}
            if "captchaState" not in captcha:
                error_key = res.get("error", {}).get("key", "UNKNOWN_ERROR")
                raise OperationError(f"Ошибка авторизации: {error_key}")

            if captcha.get("captchaError"):
                logger.warning("Капча была введена неверно!")

            payload |= self._captcha_fields(
                captcha, xsrf=xsrf, referer_url=redirect_url
            )

        raise OperationError("Не удалось пройти капчу при вводе кода")

    def _post_json(
        self,
        url: str,
        payload: dict[str, str],
        headers: dict[str, str],
        *,
        multipart: bool = False,
    ) -> dict[str, Any]:
        """POST на страницу hh с разбором JSON-ответа.

        hh на одни эндпоинты ждет multipart/form-data, на другие —
        urlencoded, поэтому формат задает вызывающий код.
        """
        session = self._tool.session
        kwargs: dict[str, Any] = (
            {"files": {k: (None, v) for k, v in payload.items()}}
            if multipart
            else {"data": payload}
        )
        r = session.post(url, headers=headers, **kwargs)
        logger.debug(f"{r.request.method} {r.url} {r.status_code}")

        try:
            return r.json()
        except json.JSONDecodeError as ex:
            raise OperationError("Неизвестная ошибка!") from ex

    def _captcha_fields(
        self,
        captcha: dict[str, Any],
        *,
        xsrf: str,
        referer_url: str,
    ) -> dict[str, str]:
        """Распознает капчу и возвращает поля для повторного запроса."""
        state = captcha["captchaState"]

        try:
            image = self._tool.get_captcha_image(
                xsrf_token=xsrf, referer_url=referer_url
            )
            text = self._tool.captcha_ai.recognize_text(image.data)
        except Exception as ex:
            raise OperationError(
                f"Не удалось распознать капчу при авторизации: {ex}"
            ) from ex

        logger.debug("Капча распознана: %r (key=%s)", text, image.key)

        if not text:
            raise OperationError("Не удалось распознать капчу при авторизации")

        return {
            "captchaText": text,
            "captchaKey": image.key,
            "captchaState": state,
        }

    @staticmethod
    def _normalize_phone_number(phone: str) -> str:
        # Удаляем всякий мусор типа (, ) и -
        phone = re.sub(r"\D", "", phone)
        if len(phone) == 11 and phone[0] == "8":
            phone = "7" + phone[1:]
        return phone

    @staticmethod
    def _login_trust_flags(
        pubkey: str,
        payload: dict[str, Any] | None = None,
    ) -> str:
        public_key = serialization.load_pem_public_key(pubkey.encode())
        payload = {
            **(payload or {}),
            "ts": int(time.time() * 1000),
        }
        logger.debug("trust login data: %r", payload)
        # JSON.stringify -> компактный вывод, без пробелов
        plaintext = json.dumps(payload, separators=(",", ":")).encode()
        ciphertext = public_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return base64.b64encode(ciphertext).decode()
