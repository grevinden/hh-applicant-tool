"""Очередь запросов к hh.ru.

Пауза между запросами раньше была вшита в три места отправки отклика
как random.uniform(1, 3), а минимальный интервал считался внутри
ApiClient. Из-за этого всё, что идёт к hh.ru мимо ApiClient — логин,
страница вакансии, капча, автоответчик, — паузу не держало.

Теперь это Throttle: одна очередь на инструмент, и все запросы к hh.ru
уходят через ThrottledSession, независимо от транспорта.

Сеть не используется: отправку подменяет session.send, а sleep
перехвачен. Мокать session.request нельзя — именно в нём живёт
ожидание, это и есть проверяемое поведение.
"""

from __future__ import annotations

import argparse
import threading
import time
from unittest.mock import MagicMock

import pytest
import requests

from hh_applicant_tool.api.client import ApiClient
from hh_applicant_tool.api.throttle import (
    DEFAULT_DELAY,
    DEFAULT_HH_RETRIES,
    DEFAULT_HH_TIMEOUT,
    DEFAULT_THROTTLE_MAX,
    DEFAULT_THROTTLE_MIN,
    Throttle,
    ThrottledSession,
    wrap_session,
)


class _FakeResponse:
    status_code = 200
    headers: dict[str, str] = {}
    url = "https://api.hh.ru/test"
    text = "{}"

    def json(self) -> dict:
        return {}

    def raise_for_status(self) -> None:
        pass


@pytest.fixture
def slept(monkeypatch) -> list[float]:
    """Список задержек вместо реального ожидания."""
    delays: list[float] = []
    monkeypatch.setattr(
        "hh_applicant_tool.api.throttle.time.sleep", delays.append
    )
    return delays


def _offline(session: requests.Session) -> requests.Session:
    """Заглушка отправки: request() и его ожидание остаются настоящими."""
    session.send = MagicMock(return_value=_FakeResponse())  # type: ignore[method-assign]
    return session


class TestThrottle:
    def test_pause_in_range(self, slept):
        """Пауза попадает в заданный диапазон."""
        Throttle(pause=(1.0, 3.0)).wait()

        assert len(slept) == 1
        assert 1.0 <= slept[0] <= 3.0

    def test_pause_differs_between_calls(self, slept):
        """Два запроса подряд не должны ждать одинаково."""
        throttle = Throttle(pause=(1.0, 3.0))

        throttle.wait()
        throttle.mark_done()
        throttle.wait()

        assert slept[0] != slept[1]

    def test_min_interval_applies_after_request(self, slept):
        """После запроса выдерживается гарантированный минимум.

        Первый вызов интервал ещё не ждёт: предыдущего запроса не
        было. Второй — уже ждёт минимум, и сверху случайную паузу.
        """
        throttle = Throttle(delay=5.0, pause=(2.0, 2.0))

        throttle.wait()
        throttle.mark_done()
        throttle.wait()

        assert slept == [2.0, pytest.approx(5.0, abs=0.5), 2.0]

    def test_zero_pause_and_zero_delay_means_no_wait(self, slept):
        """--throttle 0 0 вместе с -d 0 отключает паузу совсем."""
        throttle = Throttle(delay=0.0, pause=(0.0, 0.0))
        throttle.wait()
        throttle.mark_done()
        throttle.wait()

        assert slept == []

    def test_zero_pause_keeps_min_interval(self, slept):
        """Случайную паузу можно выключить, не теряя минимум."""
        throttle = Throttle(delay=5.0, pause=(0.0, 0.0))

        throttle.wait()
        throttle.mark_done()
        throttle.wait()

        assert slept[0] == pytest.approx(5.0, abs=0.5)

    def test_default_pause(self):
        """По умолчанию осталось то, что раньше было захардкожено."""
        throttle = Throttle()

        assert throttle.delay == DEFAULT_DELAY
        assert throttle.pause == (DEFAULT_THROTTLE_MIN, DEFAULT_THROTTLE_MAX)

    def test_callers_are_queued(self, monkeypatch):
        """Два потока не могут проскочить в сеть одновременно.

        Ожидание держится под замком, поэтому второй поток считает
        паузу уже после первого запроса. Если бы замка не было, оба
        вызова легли бы в одно и то же время.
        """
        # Ссылки надо взять до подмены: monkeypatch патчит атрибут
        # самого модуля time, то есть вообще все time.sleep
        real_sleep = time.sleep

        events: list[str] = []
        lock = threading.Lock()

        def fake_sleep(seconds: float) -> None:
            name = threading.current_thread().name
            with lock:
                events.append(f"in {name}")
            # Даём другому потоку время войти, если замка нет
            real_sleep(0.02)
            with lock:
                events.append(f"out {name}")

        monkeypatch.setattr(
            "hh_applicant_tool.api.throttle.time.sleep", fake_sleep
        )
        throttle = Throttle(delay=0.0, pause=(1.0, 1.0))

        def worker() -> None:
            throttle.wait()
            throttle.mark_done()

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        # Ожидания не переплелись: первый поток полностью вышел из
        # паузы, прежде чем второй в неё вошёл
        assert len(events) == 4
        assert events[0].startswith("in ")
        assert events[1] == events[0].replace("in ", "out ")
        assert events[2].startswith("in ")
        assert events[3] == events[2].replace("in ", "out ")

    @pytest.mark.parametrize(
        "pause",
        [
            (1.0,),
            (3.0, 1.0),
            (-1.0, 1.0),
        ],
    )
    def test_bad_pause_rejected(self, pause):
        with pytest.raises(AssertionError):
            Throttle(pause=pause)


class TestThrottledSession:
    def test_get_waits(self, slept):
        session = _offline(ThrottledSession(Throttle(pause=(1.0, 3.0))))

        session.get("https://hh.ru/vacancy/1")

        assert len(slept) == 1
        assert 1.0 <= slept[0] <= 3.0

    def test_post_waits(self, slept):
        session = _offline(ThrottledSession(Throttle(pause=(1.0, 3.0))))

        session.post("https://hh.ru/negotiations")

        assert len(slept) == 1

    def test_waits_between_requests(self, slept):
        """Главное свойство: пауза между соседними запросами."""
        session = _offline(
            ThrottledSession(Throttle(delay=0.0, pause=(1.0, 3.0)))
        )

        session.get("https://hh.ru/vacancy/1")
        session.get("https://hh.ru/vacancy/2")
        session.post("https://hh.ru/negotiations")

        assert len(slept) == 3
        assert all(1.0 <= d <= 3.0 for d in slept)

    def test_marks_done_after_request(self, slept):
        """После запроса отсчёт интервала идёт от него."""
        throttle = Throttle(delay=5.0, pause=(0.0, 0.0))
        session = _offline(ThrottledSession(throttle))

        session.get("https://hh.ru/")
        throttle.wait()

        assert slept[0] == pytest.approx(5.0, abs=0.5)

    def test_failed_request_still_marks_done(self, slept):
        """Ошибка сети не должна оставлять очередь в рассинхроне."""
        throttle = Throttle(delay=5.0, pause=(0.0, 0.0))
        session = ThrottledSession(throttle)
        session.send = MagicMock(  # type: ignore[method-assign]
            side_effect=requests.RequestException("boom")
        )

        with pytest.raises(requests.RequestException):
            session.get("https://hh.ru/")

        assert throttle._previous_request_time > 0


class TestThrottledSessionNetworkRetry:
    """Повтор запроса после обрыва/таймаута соединения с hh.ru."""

    def test_default_timeout_applied(self):
        """Без своего таймаута запрос получает общий DEFAULT_HH_TIMEOUT."""
        throttle = Throttle(pause=(0.0, 0.0))
        session = ThrottledSession(throttle)
        session.send = MagicMock(return_value=_FakeResponse())  # type: ignore[method-assign]

        session.get("https://hh.ru/vacancy/1")

        _prepared, kwargs = session.send.call_args
        assert kwargs["timeout"] == DEFAULT_HH_TIMEOUT

    def test_caller_timeout_wins(self):
        """Явный таймаут вызывающего кода не должен перетираться."""
        throttle = Throttle(pause=(0.0, 0.0))
        session = ThrottledSession(throttle)
        session.send = MagicMock(return_value=_FakeResponse())  # type: ignore[method-assign]

        session.get("https://hh.ru/vacancy/1", timeout=10)

        _prepared, kwargs = session.send.call_args
        assert kwargs["timeout"] == 10

    def test_connection_error_retried(self, slept):
        """Оборванное соединение не роняет запрос с первого раза."""
        throttle = Throttle(pause=(0.0, 0.0))
        session = ThrottledSession(throttle)
        calls: list[object] = []

        def send(prepared, **kwargs):
            calls.append(prepared)
            if len(calls) <= 1:
                raise requests.exceptions.ConnectionError("network down")
            return _FakeResponse()

        session.send = MagicMock(side_effect=send)  # type: ignore[method-assign]

        response = session.get("https://hh.ru/vacancy/1")

        assert response.status_code == 200
        assert len(calls) == 2

    def test_timeout_retried(self, slept):
        """Зависшее до таймаута соединение повторяется."""
        throttle = Throttle(pause=(0.0, 0.0))
        session = ThrottledSession(throttle)
        calls: list[object] = []

        def send(prepared, **kwargs):
            calls.append(prepared)
            if len(calls) <= 2:
                raise requests.exceptions.Timeout("timed out")
            return _FakeResponse()

        session.send = MagicMock(side_effect=send)  # type: ignore[method-assign]

        response = session.get("https://hh.ru/vacancy/1")

        assert response.status_code == 200
        assert len(calls) == 3

    def test_retries_exhausted(self, slept):
        """Когда повторы закончились, последняя ошибка всплывает."""
        throttle = Throttle(pause=(0.0, 0.0))
        session = ThrottledSession(throttle)
        calls: list[object] = []

        def send(prepared, **kwargs):
            calls.append(prepared)
            raise requests.exceptions.ConnectionError("network down")

        session.send = MagicMock(side_effect=send)  # type: ignore[method-assign]

        with pytest.raises(requests.exceptions.ConnectionError):
            session.get("https://hh.ru/vacancy/1")

        # Первая попытка плюс max_retries повторов
        assert len(calls) == DEFAULT_HH_RETRIES + 1

    def test_retries_back_off(self, slept):
        """Пауза между повторами растёт с каждой попыткой."""
        throttle = Throttle(delay=0.0, pause=(0.0, 0.0))
        session = ThrottledSession(throttle)
        session.send = MagicMock(  # type: ignore[method-assign]
            side_effect=requests.exceptions.ConnectionError("network down")
        )

        with pytest.raises(requests.exceptions.ConnectionError):
            session.get("https://hh.ru/vacancy/1")

        # Паузы троттля здесь нулевые, отсюда только сами задержки
        # повторов, и они растут с каждой попыткой
        assert slept == pytest.approx([1.0, 2.0, 3.0])

    def test_non_retryable_error_not_retried(self):
        """Не сетевой сбой повтором не лечится и не повторяется."""
        throttle = Throttle(pause=(0.0, 0.0))
        session = ThrottledSession(throttle)
        session.send = MagicMock(  # type: ignore[method-assign]
            side_effect=requests.exceptions.InvalidURL("bad url")
        )

        with pytest.raises(requests.exceptions.InvalidURL):
            session.get("https://hh.ru/vacancy/1")

        assert session.send.call_count == 1

    def test_failed_retries_still_mark_done(self, slept):
        """Очередь в синхроне даже после исчерпанных повторов."""
        throttle = Throttle(pause=(0.0, 0.0))
        session = ThrottledSession(throttle)
        session.send = MagicMock(  # type: ignore[method-assign]
            side_effect=requests.exceptions.ConnectionError("network down")
        )

        with pytest.raises(requests.exceptions.ConnectionError):
            session.get("https://hh.ru/vacancy/1")

        assert throttle._previous_request_time > 0


class TestWrapSession:
    def test_plain_session_gets_wrapped(self, slept):
        """Чужую сессию тоже надо накрыть паузой."""
        plain = requests.Session()
        plain.headers["X-Test"] = "1"
        throttle = Throttle(pause=(1.0, 3.0))

        wrapped = _offline(wrap_session(plain, throttle))
        wrapped.get("https://hh.ru/")

        assert isinstance(wrapped, ThrottledSession)
        assert wrapped.throttle is throttle
        assert wrapped.headers["X-Test"] == "1"
        assert len(slept) == 1

    def test_cookies_kept_by_reference(self):
        """Куки инструмента терять нельзя: в них авторизация."""
        plain = requests.Session()
        plain.cookies.set("hhuid", "123")

        wrapped = wrap_session(plain, Throttle())

        assert wrapped.cookies is plain.cookies
        assert wrapped.cookies.get("hhuid") == "123"

    def test_already_wrapped_keeps_queue(self):
        throttle = Throttle()
        session = ThrottledSession(throttle)

        assert wrap_session(session, throttle) is session

    def test_already_wrapped_adopts_new_queue(self):
        session = ThrottledSession(Throttle())
        new_queue = Throttle(pause=(5.0, 9.0))

        assert wrap_session(session, new_queue) is session
        assert session.throttle is new_queue


class TestApiClientUsesQueue:
    def test_creates_session_on_own(self, slept):
        """Клиент без сессии всё равно троттлит свой транспорт."""
        client = ApiClient(
            client_id="cid",
            client_secret="secret",
            access_token="USER123",
            throttle=Throttle(pause=(1.0, 3.0)),
        )
        _offline(client.session)

        client.get("/me")

        assert isinstance(client.session, ThrottledSession)
        assert len(slept) == 1
        assert 1.0 <= slept[0] <= 3.0

    def test_tool_session_and_client_share_queue(self, slept):
        """Сессия инструмента и ApiClient ждут одну очередь.

        Иначе запросы из разных мест шли бы вразнобой и пауза между
        соседними не выдерживалась бы.
        """
        throttle = Throttle(delay=0.0, pause=(1.0, 3.0))
        session = _offline(ThrottledSession(throttle))
        client = ApiClient(
            client_id="cid",
            client_secret="secret",
            access_token="USER123",
            throttle=throttle,
            session=session,
        )

        session.get("https://hh.ru/vacancy/1")
        client.post("/negotiations")

        assert client.session is session
        assert client.session.throttle is throttle  # type: ignore[union-attr]
        assert len(slept) == 2

    def test_plain_session_from_tool_gets_wrapped(self):
        """Сессия без обёртки тоже должна попасть в очередь."""
        throttle = Throttle()
        client = ApiClient(
            client_id="cid",
            client_secret="secret",
            access_token="USER123",
            throttle=throttle,
            session=requests.Session(),
        )

        assert isinstance(client.session, ThrottledSession)
        assert client.session.throttle is throttle  # type: ignore[union-attr]


class TestThrottleFlagPosition:
    """Флаг паузы должен приниматься в любом месте командной строки.

    Иначе приходится вспоминать, что глобальные флаги argparse идут
    до подкоманды, а --throttle в конце списка выглядит совершенно
    обычным флагом команды.
    """

    @pytest.mark.parametrize(
        ("argv", "expected"),
        [
            (["apply"], [1.0, 3.0]),
            (["--throttle", "5", "9", "apply"], [5.0, 9.0]),
            (["apply", "--throttle", "0.5", "1"], [0.5, 1.0]),
            (["-v", "apply", "-f", "--throttle", "0.5", "1"], [0.5, 1.0]),
            (["auth", "--throttle", "7", "8"], [7.0, 8.0]),
        ],
    )
    def test_parsed_in_any_position(self, argv, expected):
        from hh_applicant_tool.tool import HHApplicantTool

        parser = HHApplicantTool()._parser
        args = parser.parse_args(argv)

        assert args.throttle_range == expected

    def test_after_subcommand_wins(self):
        """Указанная после подкоманды пауза побеждает.

        Разбор команды в argparse копирует значения подпарсера в
        общее пространство имён, поэтому флаг после подкоманды
        затирает указанный до неё.
        """
        from hh_applicant_tool.tool import HHApplicantTool

        parser = HHApplicantTool()._parser
        args = parser.parse_args(
            ["--throttle", "5", "9", "apply", "--throttle", "0.5", "1"]
        )

        assert args.throttle_range == [0.5, 1.0]

    def test_value_before_subcommand_survives(self):
        """Значение до подкоманды не должно затираться пустым.

        Регрессия: если бы у флага в парсере команды стоял обычный
        default, подпарсер записал бы его поверх разобранного раньше.
        """
        from hh_applicant_tool.tool import HHApplicantTool

        parser = HHApplicantTool()._parser
        args = parser.parse_args(["--throttle", "5", "9", "apply"])

        assert args.throttle_range == [5.0, 9.0]


class TestToolWiring:
    def _tool(self, *argv: str):
        from hh_applicant_tool.tool import HHApplicantTool

        tool = HHApplicantTool()
        tool._assign_args(
            argparse.Namespace(**vars(tool._parser.parse_args(argv)))
        )
        return tool

    def test_one_queue_for_the_whole_tool(self):
        """Проверяем то, как это связано в main.py."""
        tool = self._tool("--throttle", "5", "9", "apply")

        assert isinstance(tool.throttle, Throttle)
        assert tool.throttle.pause == (5.0, 9.0)
        assert isinstance(tool.session, ThrottledSession)
        assert tool.session.throttle is tool.throttle
        assert tool.api_client.throttle is tool.throttle
        assert tool.api_client.session is tool.session

    def test_default_pause_applies_to_all(self):
        tool = self._tool("apply")

        assert tool.throttle.pause == (
            DEFAULT_THROTTLE_MIN,
            DEFAULT_THROTTLE_MAX,
        )

    def test_openai_session_is_not_throttled(self):
        """К OpenAI это отношения не имеет, там своя очередь."""
        tool = self._tool("apply")

        assert not isinstance(tool.openai_session, ThrottledSession)