"""Очередь запросов к hh.ru.

Раньше пауза перед откликом была вшита в три места отправки как
random.uniform(1, 3), а базовый интервал между запросами считался
внутри ApiClient. Из-за этого всё, что идёт мимо ApiClient напрямую
через сессию инструмента — логин, страница вакансии, решение капчи,
автоответчик — в очередь не попадало и паузу не держало.

Здесь живут оба интервала и держит их одна очередь на инструмент:
все запросы к hh.ru уходят через ThrottledSession, независимо от
того, отправляет их ApiClient или кто-то ещё.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Sequence
from threading import Lock
from typing import Any

import requests

logger = logging.getLogger(__package__)

# Гарантированный минимум между соседними запросами: у hh.ru стоит
# анти-DDOS
DEFAULT_DELAY = 0.345
# Случайная пауза поверх этого минимума. Значение по умолчанию — то,
# что раньше было захардкожено в трёх местах отправки отклика
DEFAULT_THROTTLE_MIN = 1.0
DEFAULT_THROTTLE_MAX = 3.0


class Throttle:
    """Пауза между запросами, общая на все транспорты инструмента.

    Объект один на инструмент, поэтому запросы выстраиваются в одну
    очередь: пауза считается от конца предыдущего запроса, и два
    параллельных вызова не могут проскочить подряд.
    """

    def __init__(
        self,
        delay: float | None = None,
        pause: Sequence[float] | None = None,
    ) -> None:
        if pause is None:
            pause = (DEFAULT_THROTTLE_MIN, DEFAULT_THROTTLE_MAX)
        else:
            pause = tuple(pause)
        assert len(pause) == 2, "pause must be [min, max]"
        assert pause[0] >= 0, "pause min must be >= 0"
        assert pause[0] <= pause[1], "pause min > max"
        self.delay = DEFAULT_DELAY if delay is None else delay
        self.pause: tuple[float, float] = (pause[0], pause[1])
        self.lock = Lock()
        self._previous_request_time = 0.0

    def wait(self) -> None:
        """Выдержать паузу перед очередным запросом."""
        with self.lock:
            # На сервере какая-то анти-DDOS система
            delay = self.delay - time.monotonic() + self._previous_request_time
            if delay > 0:
                logger.debug("wait %fs before request", delay)
                time.sleep(delay)
            pause = random.uniform(*self.pause)
            if pause:
                logger.debug("throttle %.2fs before request", pause)
                time.sleep(pause)

    def mark_done(self) -> None:
        """Отметить конец запроса: отсчёт интервала пойдёт от него."""
        with self.lock:
            self._previous_request_time = time.monotonic()


class ThrottledSession(requests.Session):
    """Сессия, которая перед каждым запросом ждёт общую паузу."""

    throttle: Throttle

    def __init__(self, throttle: Throttle | None = None) -> None:
        super().__init__()
        self.throttle = throttle if throttle is not None else Throttle()

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        self.throttle.wait()
        try:
            return super().request(method, url, **kwargs)
        finally:
            self.throttle.mark_done()


def wrap_session(
    session: requests.Session,
    throttle: Throttle,
) -> requests.Session:
    """Заменить сессию на троттлящую, сохранив её настройки.

    Куки копируются по ссылке: у инструмента это общий
    HHOnlyCookieJar с авторизацией, и терять его нельзя.
    """
    if isinstance(session, ThrottledSession):
        if session.throttle is not throttle:
            session.throttle = throttle
        return session
    wrapped = ThrottledSession(throttle)
    wrapped.cookies = session.cookies
    wrapped.headers.update(session.headers)
    wrapped.proxies = session.proxies
    wrapped.auth = session.auth
    wrapped.verify = session.verify
    wrapped.cert = session.cert
    return wrapped