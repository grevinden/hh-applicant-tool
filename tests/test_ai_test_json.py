"""Структурированный ответ на вопрос отборочного теста.

Зачем JSON, а не свободный текст: с болтовнёй («я не уверен, потому что
…») разбираться нечем, и вакансия либо уходила с мусором в поле ответа,
либо падала. Как и в капче, модель обязана вернуть JSON-объект, а
осознанный отказ выражается пустой строкой, которая разбирается в None.
"""

from __future__ import annotations

import pytest

from hh_applicant_tool.ai.openai import ChatOpenAI, OpenAIError


def client() -> ChatOpenAI:
    return ChatOpenAI(
        api_key="key",
        base_url="https://example.test/v1/chat/completions",
        model="m",
        system_prompt="пиши письма",
    )


class TestParseTestJson:
    def test_plain_json(self):
        assert (
            ChatOpenAI._parse_test_json('{"answer": "10"}') == "10"
        )

    def test_surrounding_text_is_dropped(self):
        """Модель часто пишет «Вот ответ: {...}»."""
        assert (
            ChatOpenAI._parse_test_json('Вот ответ: {"answer": "10"} — готово')
            == "10"
        )

    def test_empty_answer_means_unsure(self):
        """Осознанный отказ, а не ошибка формата."""
        assert ChatOpenAI._parse_test_json('{"answer": ""}') is None

    def test_blank_answer_means_unsure(self):
        assert ChatOpenAI._parse_test_json('{"answer": "   "}') is None

    def test_answer_is_stripped(self):
        assert ChatOpenAI._parse_test_json('{"answer": "  да  "}') == "да"

    def test_long_text_answer_allowed(self):
        assert (
            ChatOpenAI._parse_test_json(
                '{"answer": "Двенадцать лет в Python, последние пять в банке"}'
            )
            == "Двенадцать лет в Python, последние пять в банке"
        )

    def test_no_json_is_a_format_error(self):
        """Свободный текст без JSON разбирать нечем: болтовню слать нельзя."""
        with pytest.raises(OpenAIError):
            ChatOpenAI._parse_test_json("Я не уверен, что перееду")

    def test_empty_response_is_a_format_error(self):
        with pytest.raises(OpenAIError):
            ChatOpenAI._parse_test_json("")

    def test_broken_json_is_a_format_error(self):
        with pytest.raises(OpenAIError):
            ChatOpenAI._parse_test_json('{"answer": "10"')

    def test_json_array_is_a_format_error(self):
        with pytest.raises(OpenAIError):
            ChatOpenAI._parse_test_json('["10"]')

    def test_missing_key_is_a_format_error(self):
        with pytest.raises(OpenAIError):
            ChatOpenAI._parse_test_json('{"reply": "10"}')

    def test_non_string_answer_is_a_format_error(self):
        with pytest.raises(OpenAIError):
            ChatOpenAI._parse_test_json('{"answer": 10}')


class TestAnswerTestQuestion:
    def _one(self, client: ChatOpenAI, raw: str):
        client.complete = lambda prompt: raw  # type: ignore[method-assign]
        return client.answer_test_question("Вопрос?")

    def _sent_prompt(self, client: ChatOpenAI) -> str:
        """Промпт, ушедший в модель. Ответ валидный, чтобы метод
        дошёл до конца и его можно было разобрать."""
        seen: list[str] = []

        def complete(prompt: str) -> str:
            seen.append(prompt)
            return '{"answer": "да"}'

        client.complete = complete  # type: ignore[method-assign]
        client.answer_test_question("Вопрос?")
        return seen[0]

    def test_returns_parsed_answer(self):
        assert self._one(client(), '{"answer": "да"}') == "да"

    def test_unsure_gives_none(self):
        assert self._one(client(), '{"answer": ""}') is None

    def test_json_rules_come_after_the_question(self):
        """Правило формата стоит в конце: так оно перебивает сбитый
        промпт письма, который тоже попадает в то же сообщение."""
        prompt = self._sent_prompt(client())

        assert prompt.startswith("Вопрос?")
        assert "JSON object only" in prompt

    def test_letter_prompt_is_not_reused(self):
        """Промпт письма в тесте сбивает модель, поэтому тут свой."""
        prompt = self._sent_prompt(client())

        assert "сопроводительн" not in prompt.lower()