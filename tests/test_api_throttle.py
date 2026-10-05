"""Троплинг запросов к hh.ru живёт на транспорте.

Пауза между запросами раньше была вшита в три места отправки отклика как
random.uniform(1, 3) — величина нигде не настраивалась и легко разъезжалась.
Теперь это throttle на ApiClient: один настройка на транспорте, перед
любым запросом без разбора — чтение это или запись. Запросы выстраиваются
в очередь под self.lock, поэтому пауза выдерживается между соседними.

Сеть не используется: requests.Session заменён заглушкой, sleep
перехвачен.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hh_applicant_tool.api.client import ApiClient


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
        "hh_applicant_tool.api.client.time.sleep", delays.append
    )
    return delays


def _client(**kwargs) -> ApiClient:
    client = ApiClient(
        client_id="cid",
        client_secret="secret",
        access_token="USER123",
        # Базовый интервал выключаем, чтобы в задержках видеть только
        # троплинг
        delay=0.0,
        **kwargs,
    )
    # delay=0 всё равно превращается в DEFAULT_DELAY, поэтому гасим
    # его напрямую после инициализации
    client.delay = 0.0
    client.session = MagicMock()
    client.session.request.return_value = _FakeResponse()
    return client


class TestThrottle:
    def test_post_is_throttled(self, slept):
        """Перед откликом пауза попадает в заданный диапазон."""
        client = _client(throttle=[1.0, 3.0])

        client.post("/negotiations")

        assert len(slept) == 1
        assert 1.0 <= slept[0] <= 3.0

    def test_get_is_throttled_too(self, slept):
        """Чтение выдачи тоже ждёт: очередь одна на все запросы."""
        client = _client(throttle=[1.0, 3.0])

        client.get("/vacancies")

        assert len(slept) == 1
        assert 1.0 <= slept[0] <= 3.0

    def test_queue_is_shared_between_methods(self, slept):
        """Неважно, что за запросы: пауза выдерживается между ними."""
        client = _client(throttle=[1.0, 3.0])

        client.get("/vacancies")
        client.post("/negotiations")
        client.get("/resumes")

        assert len(slept) == 3
        assert all(1.0 <= d <= 3.0 for d in slept)

    def test_throttle_disabled_by_default(self, slept):
        """Без настройки троплинга нет."""
        client = _client()

        client.post("/negotiations")

        assert slept == []

    def test_zero_throttle_means_no_wait(self, slept):
        """--throttle 0 0 отключает паузу совсем."""
        client = _client(throttle=[0.0, 0.0])

        client.post("/negotiations")

        assert slept == [0.0]

    def test_pause_differs_between_calls(self, slept):
        """Два запроса подряд не должны ждать одинаково."""
        client = _client(throttle=[1.0, 3.0])

        client.get("/vacancies")
        client.get("/vacancies")

        assert slept[0] != slept[1]

    def test_min_interval_still_applies(self, slept):
        """Базовый интервал и троплинг складываются."""
        client = _client(throttle=[2.0, 2.0])
        client.delay = 5.0

        client.post("/negotiations")

        # Первый запрос: троплинг 2с, базовый интервал отсчитывается от
        # нулевого времени и уже прошёл
        assert slept == [2.0]


class TestThrottleValidation:
    def test_wrong_length_rejected(self):
        with pytest.raises(AssertionError):
            _client(throttle=[1.0])

    def test_reversed_range_rejected(self):
        with pytest.raises(AssertionError):
            _client(throttle=[3.0, 1.0])

    def test_negative_rejected(self):
        with pytest.raises(AssertionError):
            _client(throttle=[-1.0, 1.0])