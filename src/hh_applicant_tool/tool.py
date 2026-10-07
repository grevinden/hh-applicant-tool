from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
import signal
import smtplib
import sqlite3
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from http.cookiejar import CookieJar, MozillaCookieJar
from importlib import import_module
from itertools import count
from os import getenv
from pathlib import Path
from pkgutil import iter_modules
from typing import Any, Callable, Iterable, Type, TypedDict
from urllib.parse import parse_qsl, urljoin, urlsplit

import requests

from . import ai, api, utils
from .api.throttle import (
    DEFAULT_THROTTLE_MAX,
    DEFAULT_THROTTLE_MIN,
    Throttle,
    ThrottledSession,
)
from .constants import (
    CONFIG_DIR,
    CONFIG_FILENAME,
    COOKIES_FILENAME,
    DATABASE_FILENAME,
    DEFAULT_CAPTCHA_LANGUAGE,
    DEFAULT_OPENAI_CONNECT_TIMEOUT,
    DEFAULT_OPENAI_TIMEOUT,
    DEFAULT_SITE_LANGUAGE,
    DESKTOP_USER_AGENT,
    LOG_FILENAME,
)
from .storage import StorageFacade
from .utils.argparse import ArgumentFormatter
from .utils.cookiejar import HHOnlyCookieJar
from .utils.log import setup_logger
from .utils.mixins import MegaTool
from .utils.terminal import print_kitty_image, print_sixel_image

logger = logging.getLogger(__package__)

OPERATIONS = "operations"


class HHLuxInitialState(TypedDict):
    redirectConfig: dict[str, Any]
    ...


class CaptchaInfo:
    url: str
    key: str
    url: str
    lang: str
    image_data: bytes


class Error(Exception):
    pass


class BaseOperation:
    def setup_parser(self, parser: argparse.ArgumentParser) -> None: ...

    def run(
        self,
        tool: HHApplicantTool,
        args: BaseNamespace,  # pyright: ignore[reportUnusedParameter]
    ) -> None | int:
        raise NotImplementedError()


class HHSession(requests.Session):
    cookies: HHOnlyCookieJar


class BaseAttrs:
    profile_id: str
    config_dir: Path
    verbosity: int
    api_delay: float
    throttle_range: list[float] | None
    user_agent: str
    proxy_url: str
    use_sixel: bool
    use_kitty: bool
    manual: bool
    captcha_lang: str
    captcha_attempts: int
    openai_proxy_url: str
    openai_timeout: float
    openai_connect_timeout: float


class BaseNamespace(argparse.Namespace, BaseAttrs):
    operation_run: Callable[[HHApplicantTool, BaseNamespace], None | int] | None


@dataclass
class BaseAPICaptchaHandler(ABC):
    tool: HHApplicantTool

    @abstractmethod
    def __call__(self, captcha_url: str) -> bool:
        """Этот метод обязан переопределить каждый наследник."""
        pass


class APICaptchaHandler(BaseAPICaptchaHandler):
    def __call__(self, captcha_url: str) -> bool:
        if self.tool.manual:
            return self.tool.solve_captcha_manual(captcha_url)

        solver = self.tool.captcha_solver
        if solver is None:
            # Ещё никто не подключил свой путь решения: возвращаем
            # False, и ApiClient пробрасывает CaptchaRequired, как раньше
            return False
        return bool(solver(captcha_url))


class HHApplicantTool(MegaTool, BaseAttrs):
    """Утилита для автоматизации действий соискателя на сайте hh.ru.

    Исходники и предложения: <https://github.com/s3rgeym/hh-applicant-tool>

    Группа поддержки: <https://t.me/s3rgeym_chat>
    """

    # Чем решать капчу, пришедшую на любой запрос ApiClient, а не
    # только на отклик. Региструет та операция, у которой пайплайн
    # есть: пока здесь None, хендлер отвечает отказом и ApiClient
    # кидает CaptchaRequired, как до этой связи
    captcha_solver: Callable[[str], bool] | None = None

    @staticmethod
    def _add_throttle_argument(
        parser: argparse.ArgumentParser,
        *,
        default: Any,
    ) -> None:
        """Флаг случайной паузы между запросами к HH.

        В парсерах команд default=SUPPRESS, а не значение: argparse
        разбирает команду в отдельном пространстве имён и копирует
        оттуда все ключи в общее, так что обычный default затирал бы
        паузу, указанную до подкоманды.
        """
        parser.add_argument(
            "--throttle",
            nargs=2,
            type=float,
            metavar=("MIN", "MAX"),
            dest="throttle_range",
            default=default,
            help="Случайная пауза между запросами к HH: MIN и MAX секунд. И API, и логин, и страницы, и капча идут через одну очередь.",
        )

    @classmethod
    def _create_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=cls.__doc__,
            formatter_class=ArgumentFormatter,
        )
        parser.add_argument(
            "-v",
            "--verbosity",
            help="При использовании от одного и более раз увеличивает количество отладочной информации в выводе",  # noqa: E501
            action="count",
            default=0,
        )
        parser.add_argument(
            "-c",
            "--config-dir",
            "--config",
            help="Путь до директории с конфигом",
            type=Path,
            default=None,
        )
        parser.add_argument(
            "--profile-id",
            "--profile",
            help="Используемый профиль — подкаталог в --config-dir. Так же можно передать через переменную окружения HH_PROFILE_ID.",
        )
        parser.add_argument(
            "-d",
            "--api-delay",
            "--delay",
            type=float,
            help="Задержка между запросами к API HH по умолчанию",
        )
        cls._add_throttle_argument(
            parser, default=[DEFAULT_THROTTLE_MIN, DEFAULT_THROTTLE_MAX]
        )
        parser.add_argument(
            "--user-agent",
            help="User-Agent для каждого запроса",
        )
        parser.add_argument(
            "--proxy-url",
            help="Прокси, используемый для запросов и авторизации",
        )
        parser.add_argument(
            "--openai-proxy",
            "--ai-proxy",
            dest="openai_proxy_url",
            help="Отдельный прокси, используемый только для OpenAI чата",
        )
        parser.add_argument(
            "--openai-timeout",
            "--ai-timeout",
            type=float,
            help="Таймаут запроса к OpenAI в секундах: соединение и чтение ответа",
        )
        parser.add_argument(
            "--openai-connect-timeout",
            "--ai-connect-timeout",
            type=float,
            help="Таймаут соединения с OpenAI в секундах",
        )
        parser.add_argument(
            "-m",
            "--manual",
            action="store_true",
            help="Ручной режим ввода (капчи)",
        )
        parser.add_argument(
            "-k",
            "--use-kitty",
            "--kitty",
            action="store_true",
            help="Вывод капчи в kitty",
        )
        parser.add_argument(
            "-s",
            "--use-sixel",
            "--sixel",
            action="store_true",
            help="Вывод капчи в sixel",
        )
        parser.add_argument(
            "--captcha-lang",
            default=DEFAULT_CAPTCHA_LANGUAGE,
            help="Язык капчи. Некоторые модели распознают лучше текст на английском",
        )
        parser.add_argument(
            "--captcha-attempts",
            default=3,
            help="Максимальное количество неудачных попыток автоматического распознания капчи",
        )
        subparsers = parser.add_subparsers(help="commands")
        package_dir = Path(__file__).resolve().parent / OPERATIONS
        for _, module_name, _ in iter_modules([str(package_dir)]):
            if module_name.startswith("_"):
                continue
            mod = import_module(f"{__package__}.{OPERATIONS}.{module_name}")
            op: BaseOperation = mod.Operation()
            kebab_name = module_name.replace("_", "-")
            op_parser = subparsers.add_parser(
                kebab_name,
                aliases=getattr(op, "__aliases__", []),
                description=op.__doc__,
                formatter_class=ArgumentFormatter,
            )
            op_parser.set_defaults(operation_run=op.run)
            op.setup_parser(op_parser)
            # Флаги продублированы в командах, чтобы их можно было писать
            # в любом месте командной строки, а не только до подкоманды
            cls._add_throttle_argument(
                op_parser, default=argparse.SUPPRESS
            )
        parser.set_defaults(operation_run=None)
        return parser

    def __init__(
        self,
        *,
        captcha_handler_class: Type[BaseAPICaptchaHandler] | None = None,
    ):
        self._parser = self._create_parser()
        self._captcha_handler_class = captcha_handler_class

    @staticmethod
    def _proxy_url_to_dict(proxy_url: str | None) -> dict[str, str]:
        if not proxy_url:
            return {}

        return {
            "http": proxy_url,
            "https": proxy_url,
        }

    def _get_proxies(self) -> dict[str, str]:
        proxy_url = self.proxy_url or self.config.get("proxy_url")

        if proxy_url:
            return self._proxy_url_to_dict(proxy_url)

        proxies = {}
        http_env = getenv("HTTP_PROXY") or getenv("http_proxy")
        https_env = getenv("HTTPS_PROXY") or getenv("https_proxy") or http_env

        if http_env:
            proxies["http"] = http_env
        if https_env:
            proxies["https"] = https_env

        return proxies

    def _get_openai_proxies(self) -> dict[str, str]:
        openai_config = self.config.get("openai", {})
        proxy_url = self.openai_proxy_url or openai_config.get("proxy_url")
        if proxy_url:
            return self._proxy_url_to_dict(proxy_url)
        return self._get_proxies()

    def _create_http_session(
        self,
        proxies: dict[str, str],
        *,
        log_label: str,
        throttle: Throttle | None = None,
    ) -> requests.Session:
        # Троттлящая сессия — это способ накрыть троплингом всё, что
        # идёт к hh.ru: и ApiClient, и прямые обращения вроде логина
        # или страницы вакансии. К OpenAI это отношения не имеет, там
        # своя очередь в ai/openai.py, поэтому throttle там не
        # передаётся
        session: requests.Session = (
            ThrottledSession(throttle) if throttle else requests.Session()
        )

        if proxies:
            logger.info("Use proxies for %s: %r", log_label, proxies)
            session.proxies = proxies

        session.headers.update({"User-Agent": DESKTOP_USER_AGENT})
        return session

    @cached_property
    def throttle(self) -> Throttle:
        """Очередь запросов к hh.ru, общая для всех транспортов.

        Пауза между соседними запросами и гарантированный минимум
        задаются здесь и больше нигде: ни в логике отправки отклика,
        ни в коде автоответчика или капчи.
        """
        config = self.config
        return Throttle(
            delay=self.api_delay or config.get("api_delay"),
            pause=self.throttle_range or config.get("throttle"),
        )

    @cached_property
    def session(self) -> HHSession:
        session = self._create_http_session(
            self._get_proxies(),
            log_label="requests",
            throttle=self.throttle,
        )

        session.cookies = HHOnlyCookieJar(str(self.cookies_file))
        if self.cookies_file.exists():
            session.cookies.load(ignore_discard=True, ignore_expires=True)

        # Язык сайта нужен именно в момент запроса отклика: hh.ru
        # фиксирует язык картинки капчи, когда выдает captcha_url,
        # поэтому кука должна быть в сессии заранее
        self._apply_site_language(session.cookies)

        return session

    def _apply_site_language(self, jar: CookieJar) -> None:
        """Просит hh.ru отдавать сайт на нужном языке.

        Язык берется из конфигурации (site_language), по умолчанию
        английский. На язык картинки капчи эта кука не действует:
        скрипт задаёт параметр lang у POST /captcha, см. api/captcha.py.
        Пустое значение в конфиге отключает подмену, чтобы hh.ru сам
        выбрал язык (например, когда в аккаунте его уже переключили).
        """
        language = self.config.get("site_language", DEFAULT_SITE_LANGUAGE)
        language = (language or "").strip()

        if not language:
            logger.debug(
                "site_language в конфиге пустой, язык сайта не меняю",
            )
            return

        if not jar.set_site_language(language):
            logger.warning(
                "Не удалось выставить язык сайта %s, hh.ru может "
                "отдать капту на языке аккаунта",
                language,
            )
            return

        logger.info("Язык сайта hh.ru: %s", language)

    @cached_property
    def openai_session(self) -> requests.Session:
        return self._create_http_session(
            self._get_openai_proxies(),
            log_label="OpenAI requests",
        )

    @cached_property
    def config_path(self) -> Path:
        return (
            (self.config_dir or Path(getenv("CONFIG_DIR", CONFIG_DIR)))
            / (self.profile_id or getenv("HH_PROFILE_ID", "."))
        ).resolve()

    @cached_property
    def config(self) -> utils.Config:
        return utils.Config(self.config_path / CONFIG_FILENAME)

    @cached_property
    def log_file(self) -> Path:
        return self.config_path / LOG_FILENAME

    @cached_property
    def cookies_file(self) -> Path:
        return self.config_path / COOKIES_FILENAME

    @cached_property
    def db_path(self) -> Path:
        return self.config_path / DATABASE_FILENAME

    @cached_property
    def db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return conn

    @cached_property
    def storage(self) -> StorageFacade:
        return StorageFacade(self.db)

    @cached_property
    def api_client(self) -> api.client.ApiClient:
        config = self.config
        token = config.get("token", {})
        return api.client.ApiClient(
            client_id=config.get("client_id"),
            client_secret=config.get("client_secret"),
            access_token=token.get("access_token"),
            refresh_token=token.get("refresh_token"),
            access_expires_at=token.get("access_expires_at"),
            throttle=self.throttle,
            user_agent=self.user_agent or config.get("user_agent"),
            session=self.session,
            captcha_handler=self._captcha_handler_class(self)
            if self._captcha_handler_class
            else None,
        )

    def get_me(self) -> api.datatypes.User:
        return self.api_client.get("/me")

    def get_resumes(self) -> list[api.datatypes.Resume]:
        return self.api_client.get("/resumes/mine").get("items", [])

    def first_resume_id(self) -> str:
        resume = self.get_resumes()[0]
        return resume["id"]

    def get_blacklisted(self) -> list[str]:
        rv = []
        for page in count():
            r: api.datatypes.PaginatedItems[api.datatypes.EmployerShort] = (
                self.api_client.get("/employers/blacklisted", page=page)
            )
            rv += [item["id"] for item in r["items"]]
            if page + 1 >= r["pages"]:
                break
        return rv

    def get_negotiations(
        self, status: str = "active"
    ) -> Iterable[api.datatypes.Negotiation]:
        for page in count():
            r: dict[str, Any] = self.api_client.get(
                "/negotiations",
                page=page,
                per_page=100,
                status=status,
            )

            items = r.get("items", [])

            if not items:
                break

            yield from items

            if page + 1 >= r.get("pages", 0):
                break

    def _is_authenticated(self, config: dict[str, Any]) -> bool:
        account = config.get("account") or {}
        if not account:
            return False
        # Если пользователь неавторизован содержит поля типа firstName, lastName и тд со значением None (все поля)
        return any(v is not None for v in account.values())

    def parse_initial_state(
        self, response: requests.Response, check_auth: bool = True
    ) -> HHLuxInitialState:
        """Возвращает декодированное содержимое <template id="HH-Lux-InitialState"></template>"""
        if response.status_code != 200:
            raise Error(
                f"Неожиданный код ответа: {response.status_code} {response.url}"
            )

        try:
            raw_data = response.text.split('id="HH-Lux-InitialState">')[
                1
            ].split("</template>")[0]
        except IndexError as ex:
            raise Error(
                f"Template with initial state data not found on {response.url}"
            ) from ex

        # Теперь кавычки всегда превращаются в сущности?
        if raw_data.startswith("{&#34;"):
            raw_data = html.unescape(raw_data)

        # import tempfile
        # with tempfile.NamedTemporaryFile('w', delete=False, prefix='hh_initial_state_', suffix='.json', dir='.', encoding='utf-8') as tmp_file:
        #     tmp_file.write(raw_data)
        #     file_path = tmp_file.name
        #     print(file_path)

        data = json.loads(raw_data)
        assert type(data) is dict
        assert "redirectConfig" in data
        if check_auth and not self._is_authenticated(data):
            raise Error("Авторизация истекла требуется новая!")

        return data

    def fetch_initial_state(
        self, url: str, check_auth: bool = True
    ) -> HHLuxInitialState:
        return self.parse_initial_state(self.session.get(url), check_auth)

    # TODO: добавить еще методов или те удалить?

    def save_token(self) -> bool:
        if self.api_client.access_token != self.config.get("token", {}).get(
            "access_token"
        ):
            self.config.save(token=self.api_client.get_access_token())
            return True
        return False

    def save_cookies(self) -> None:
        """Сохраняет текущие куки сессии в файл."""
        if isinstance(self.session.cookies, MozillaCookieJar):
            self.session.cookies.save(ignore_discard=True, ignore_expires=True)
            logger.debug("Cookies saved to %s", self.cookies_file)
        else:
            logger.warning(
                f"Сессионные куки имеют неправильный тип: {type(self.session.cookies)}"
            )

    def get_cover_letter_ai(self, system_prompt: str) -> ai.ChatOpenAI:
        return self.get_ai_client(system_prompt, purpose="cover_letter")

    def get_vacancy_filter_ai(self, system_prompt: str) -> ai.ChatOpenAI:
        return self.get_ai_client(system_prompt, purpose="vacancy_filter")

    def get_chat_ai(self, system_prompt: str) -> ai.ChatOpenAI:
        return self.get_ai_client(system_prompt, purpose="chat")

    def get_test_ai(self) -> ai.ChatOpenAI:
        # Промпт тут свой: вопросы отборочного теста — не сопроводительное
        # письмо, и промпт письма модель только сбивает. Раньше тестовые
        # вопросы уходили тем же клиентом, что и письмо, поэтому модель
        # писала письмо там, где от неё ждали «да» или номер варианта.
        return self.get_ai_client(
            system_prompt=(
                "Ты отвечаешь на вопросы отборочного теста вакансии на hh.ru. "
                "Отвечай кратко, по делу и на языке вопроса, без вступлений "
                "и без рассуждений вслух. Ничего не выдумывай о соискателе: "
                "если данных не хватает, ответь нейтрально и обтекаемо."
            ),
            purpose="test",
        )

    def get_captcha_ai(self) -> ai.ChatOpenAI:
        return self.get_ai_client(
            system_prompt=(
                "You read CAPTCHA images. Return ONLY the text from the "
                "image, exactly as it is written."
            ),
            purpose="captcha",
        )

    def get_ai_client(
        self,
        system_prompt: str,
        purpose: str | None = None,
    ) -> ai.ChatOpenAI:
        config_sections = {
            "cover_letter": "openai_cover_letter",
            "vacancy_filter": "openai_vacancy_filter",
            "captcha": "openai_captcha",
            "chat": "openai_chat",
            "test": "openai_test",
        }

        # Отдельный раздел для тестов заводить необязательно: пока его
        # нет, вопросы теста ходят туда же, куда письма, но со своим
        # промптом. Так не появляется новый адрес шлюза, который надо
        # ещё прописать, чтобы просто ответить на тест.
        fallback_purposes = {"test": "cover_letter"}

        c = self.config.get("openai", {})

        if purpose is not None:
            if purpose not in config_sections:
                raise ValueError(
                    f"Неизвестная цель AI: {purpose}. "
                    f"Допустимые значения: {list(config_sections.keys())}"
                )

            if not self.config.get(config_sections[purpose], {}):
                purpose = fallback_purposes.get(purpose, purpose)

            purpose_config = self.config.get(config_sections[purpose], {})
            # Переписываем значения openai
            c = {**c, **purpose_config}

        api_key = c.get("api_key")
        if not api_key:
            raise ValueError(
                "API-ключ не задан. Укажите 'api_key' в секции 'openai'"
                + (f" или '{config_sections[purpose]}'." if purpose else ".")
            )

        base_url = c.get("base_url")
        if not base_url:
            raise ValueError(
                "Параметр 'base_url' не задан. Укажите его в секции 'openai'"
                + (f" или '{config_sections[purpose]}'." if purpose else ".")
            )

        model = c.get("model")
        if not model:
            logger.warning(
                "Параметр 'model' не задан в конфигурации."
                + (
                    f" Секции 'openai' и '{config_sections[purpose]}' не содержат "
                    "этого параметра."
                    if purpose
                    else " Секция 'openai' не содержит этого параметра."
                )
            )

        # Роль системного промпта задаёт модель в конфиге, а не флаг
        # запуска: настроенную модель незачем переключать перед
        # каждым запуском. system — обычное поведение, developer — для
        # шлюза с агентом, всё остальное — без системного сообщения.
        # Разрешает значение сам клиент в __post_init__: разрешить его
        # здесь тоже нельзя, иначе «без системного сообщения» не
        # отличить от «не задано»
        system_role = c.get("system_role")

        # Заголовки из секции openai и из секции цели дописываются друг к
        # другу, а не заменяют друг друга: иначе openai_captcha молча
        # убирала бы общий ключ шлюза. Обычное слияние c выше здесь не
        # годится — оно перекрывает весь extra_headers.
        #
        # Имена не выпиливаются и не перекрываются: заголовки идут
        # мультисетом, поэтому одноимённые уходят оба, и какой считать
        # своим решает шлюз. Порядок общий, затем целевой.
        extra_headers = ai.normalize_headers(
            self.config.get("openai", {}).get("extra_headers")
        ) + (
            ai.normalize_headers(
                self.config.get(
                    config_sections[purpose], {}
                ).get("extra_headers")
            )
            if purpose is not None
            else []
        )

        return ai.ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=c.get("temperature", 0.0),
            max_completion_tokens=c.get("max_completion_tokens", 1000),
            system_prompt=system_prompt,
            base_url=base_url,
            extra_headers=extra_headers or None,
            system_role=system_role,
            rate_limit=c.get("rate_limit", 40),
            timeout=(
                self.openai_timeout
                or c.get("timeout")
                or DEFAULT_OPENAI_TIMEOUT
            ),
            connect_timeout=(
                self.openai_connect_timeout
                or c.get("connect_timeout")
                or DEFAULT_OPENAI_CONNECT_TIMEOUT
            ),
            session=self.openai_session,
        )

    # TODO: вынести в миксин какой
    def get_cookie(self, name: str) -> str | None:
        """Значение cookie по имени из jar на базе {CookieJar} (нет get_dict)."""
        return next(
            (c.value for c in self.session.cookies if c.name == name),
            None,
        )

    def _extract_xsrf_token(self, content: str) -> str:
        # hh.ru отдает этот блок с HTML-заэкранированными кавычками
        # (внутри HTML-атрибута), поэтому сначала разэкранируем всю страницу
        content = html.unescape(content)
        tokens = re.findall(r',"xsrfToken":"([^"]+)"', content)
        if not tokens:
            raise ValueError("xsrf token not found")

        # На странице hh.ru может быть несколько xsrfToken. Первый из них —
        # случайное значение, которое ротируется при каждой загрузке и НЕ
        # соответствует cookie `_xsrf`, из-за чего POST на
        # /applicant/vacancy_response/popup возвращал 403 (CSRF mismatch).
        # Сервер сверяет токен именно с cookie `_xsrf`, поэтому отдаем
        # совпадающее значение, а не первое вхождение.
        cookie_xsrf = self.get_cookie("_xsrf")
        if cookie_xsrf and cookie_xsrf in tokens:
            return cookie_xsrf
        return tokens[0]

    def _get_xsrf_token(self, url: str | None = None) -> str:
        """Возвращает XSRF-токен, который выдается на сессию."""
        # Токен, который сервер реально валидирует, лежит в cookie `_xsrf`.
        # Если cookie уже есть — используем его и не делаем лишний GET.
        cookie_xsrf = self.get_cookie("_xsrf")
        if cookie_xsrf:
            return cookie_xsrf
        r = self.session.get(url or "https://hh.ru/")
        return self._extract_xsrf_token(r.text)

    @cached_property
    def xsrf_token(self) -> str:
        return self._get_xsrf_token()

    @property
    def is_logged_in(self) -> bool:
        """Проверяет авторизован ли пользователь через сайт."""
        return self.session.get("https://hh.ru/settings").status_code == 200

    @cached_property
    def smtp(self) -> smtplib.SMTP | smtplib.SMTP_SSL:
        conf = self.config.get("smtp", {})
        host = conf.get("host")
        port = conf.get("port")
        user = conf.get("user")
        password = conf.get("password")
        use_ssl = conf.get("ssl", False)

        if not host or not port:
            raise ValueError("SMTP host or port not configured")

        client_cls = smtplib.SMTP_SSL if use_ssl else smtplib.SMTP
        server = client_cls(host, port)

        if not use_ssl and conf.get("starttls", True):
            server.starttls()

        if user and password:
            server.login(user, password)

        return server

    def _fetch_captcha(
        self, captcha_url: str, lang: str = DEFAULT_CAPTCHA_LANGUAGE
    ) -> CaptchaInfo:
        """Получает изображение в виде набора байт. Вторым аргументом можно передать язык"""
        captcha_state = dict(parse_qsl(urlsplit(captcha_url).query))["state"]

        logger.debug("Получаем куки со страницы: %s", captcha_url)
        # Предполагаю, что на этой странице кука какая-то ставится
        r = self.session.get(captcha_url)
        r.raise_for_status()

        # Тут пока ничего не нужно как заглушка используется
        data = self.parse_initial_state(r)
        logger.debug("Initial State Keys:  %s", ", ".join(*data))
        assert data["hhcaptcha"]["captchaState"] == captcha_state

        # Страница, где каптча показывается
        # Обычно редиректит на страницу города
        referer_url = r.url

        # Потом кука используется для получения captcha key
        captcha_key_url = urljoin(referer_url, "/captcha?lang=" + lang)
        logger.debug(
            "Отправляем POST-запрос на %s для получения captcha key",
            captcha_key_url,
        )
        js = self.session.post(
            captcha_key_url,
            headers={
                "Referer": referer_url,
                "X-Xsrftoken": self.xsrf_token,
                "x-hhtmfrom": "",
                "x-hhtmsource": "account_captcha",
                # Я тут опустил кучу заголовков, так как их значения есть в
                # кукис, и сайт, если тех нет, берех их от туда
                # Те запрос проходит
                "X-Requested-With": "XMLHttpRequest",
            },
        ).json()

        captcha_key = js["key"]

        captcha_image_url = urljoin(
            referer_url, "/captcha/picture?key=" + captcha_key
        )

        logger.debug("Пробуем загрузить каптчу: %s", captcha_image_url)
        # А с помощью captcha key получаем изображение
        captcha_image_data = self.session.get(
            captcha_image_url, headers={"Referer": referer_url}
        ).content

        assert len(captcha_image_data) > 0, "Ошибка загрузки изображения"

        return {
            "key": captcha_key,
            "image_data": captcha_image_data,
            "state": captcha_state,
            "url": referer_url,
            "lang": lang,
        }

    def _send_captcha(self, url: str, text: str, key: str, state: str) -> bool:
        target_url: str = urljoin(url, "/account/captcha")

        payload = {
            "captchaText": text,
            "captchaKey": key,
            "captchaState": state,
            # Я не уверен, что эти параметры обзяательные
            "backurl": "/",
            "fialurl": target_url + "?state=" + state,
        }

        # Там зачем-то payload передается и в теле запроса и в query string
        # Скорее всего его можно передать только в теле
        r = self.session.post(
            target_url,
            params=payload,
            data=payload,
            headers={
                "Referer": url,
                "X-Requested-With": "XMLHttpRequest",
                "X-Xsrftoken": self.xsrf_token,
                "x-hhtmfrom": "",
                "x-hhtmsource": "account_captcha",
            },
        )

        logger.debug(
            "Код ответа сервера на отправку текста каптчи: %d", r.status_code
        )
        return r.status_code == 200

    def solve_captcha_manual(self, captcha_url: str) -> bool:
        assert self.use_kitty or self.use_sixel, (
            "Для ручного решения каптчи нужно использовать один из флагов: --use-sixel/--use-kitty"
        )
        try:
            while True:
                captcha = self._fetch_captcha(captcha_url, self.captcha_lang)
                if self.use_kitty:
                    print_kitty_image(captcha["image_data"])
                else:
                    print_sixel_image(captcha["image_data"])
                text = input("Введите текст с картинки выше: ")
                if self._send_captcha(
                    captcha["url"], text, captcha["key"], captcha["state"]
                ):
                    return True
                print("Попробуй еще!")
        except (KeyboardInterrupt, EOFError):
            return False

    @cached_property
    def captcha_ai(self) -> ai.ChatOpenAI:
        return self.get_captcha_ai()

    def solve_captcha_ai(self, captcha_url: str) -> bool:
        for attempt in range(1, self.captcha_attempts + 1):
            logger.debug(
                "(%d/%d) try to solve captcha: %s",
                attempt,
                self.captcha_attempts,
                captcha_url,
            )
            captcha = self._fetch_captcha(captcha_url, self.captcha_lang)
            text = self.captcha_ai.recognize_text(
                captcha["image_data"],
                language=self.captcha_lang,
            )
            logger.debug("AI answer for %s: %s", captcha_url, text)
            if self._send_captcha(
                captcha["url"], text, captcha["key"], captcha["state"]
            ):
                logger.debug("Captcha accepted for %s", captcha_url)
                return True
        logger.warning("Can't solve captcha for %s", captcha_url)
        return False

    def run(self, argv: Sequence[str] | None = None) -> None | int:
        args = self._parser.parse_args(argv, namespace=BaseNamespace())
        self._assign_args(args)

        # Создаем путь до конфига
        self.config_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        verbosity_level = max(
            logging.DEBUG,
            logging.WARNING - self.verbosity * 10,
        )

        setup_logger(logger, verbosity_level, self.log_file)

        logger.debug("Путь до профиля: %s", self.config_path)

        utils.setup_terminal()

        try:
            with self._graceful_sigint(args):
                if not self.operation_run:
                    self._parser.print_help(file=sys.stderr)
                    return 2
                return self._run_operation(args)
        finally:
            self._check_system()

    @contextmanager
    def _graceful_sigint(self, args: BaseNamespace):
        """Мягкое прерывание по Ctrl+C (SIGINT).

        Первое нажатие останавливает операцию между шагами (через
        `_cancel_event`), второе — принудительно завершает процесс с кодом 130.
        Ручной обработчик нужен, чтобы KeyboardInterrupt не превращался в дамп
        стека внутри сетевого вызова проверки версии в `finally` (там ловится
        только `Exception`).
        """
        cancel_event = threading.Event()
        op_instance = (
            getattr(self.operation_run, "__self__", None)
            if self.operation_run
            else None
        )
        if op_instance is not None:
            op_instance._cancel_event = cancel_event
        args._cancel_event = cancel_event

        sigint_count = [0]

        def _handle_sigint(signum, frame):
            sigint_count[0] += 1
            if sigint_count[0] == 1:
                cancel_event.set()
                logger.warning(
                    "Выполнение прервано пользователем! Приступаю к Завершению работы. "
                    "Нажмите ещё раз для принудительного выхода."
                )
            else:
                sys.exit(130)

        previous_handler = signal.signal(signal.SIGINT, _handle_sigint)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, previous_handler)

    def _run_operation(self, args: BaseNamespace) -> None | int:
        """Запускает выбранную операцию и превращает исключения в сообщения."""
        try:
            return self.operation_run(self, args)
        except KeyboardInterrupt:
            logger.warning("Выполнение прервано пользователем!")
        except api.errors.CaptchaRequired as ex:
            logger.error(f"Требуется ввод капчи: {ex.captcha_url}")
        except api.errors.InternalServerError:
            logger.error(
                "Сервер HH.RU не смог обработать запрос из-за высокой"
                " нагрузки или по иной причине"
            )
        except api.errors.Forbidden:
            logger.error("Требуется авторизация")
        except (Error, ValueError) as ex:
            logger.error(ex)
        except sqlite3.Error as ex:
            logger.exception(ex)

            script_name = sys.argv[0].split(os.sep)[-1]

            logger.warning(
                f"Возможно база данных повреждена, попробуйте выполнить команду:\n\n"  # noqa: E501
                f"  {script_name} migrate-db"
            )
        except Exception as e:
            logger.exception(e)
        finally:
            # Токен мог автоматически обновиться
            if self.save_token():
                logger.info("Токен был сохранен после обновления.")

            try:
                self.save_cookies()
            except Exception as ex:
                logger.error(f"Не удалось сохранить cookies: {ex}")
        return 1

    def _assign_args(self, args: BaseNamespace) -> None:
        for name, value in vars(args).items():
            setattr(self, name, value)
