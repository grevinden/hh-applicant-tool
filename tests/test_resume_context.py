"""Разделы резюме и вакансии, которые уходят в промпт модели.

Зачем это: в живом запуске в промпте оказывались разделы «О СЕБЕ» и
«ОПЫТ РАБОТЫ» без содержимого, потому что hh.ru отдаёт эти поля пустыми,
а заголовки печатались безусловно. Заодно терялись уже имеющиеся данные —
образование, языки, роли, город, зарплата, — хотя они помогают модели
писать письмо.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from hh_applicant_tool.operations.apply_vacancies import Operation

FULL_RESUME = {
    "id": "resume-1",
    "title": "Системный администратор Linux",
    "skills": None,
    "skill_set": ["Linux", "Ansible"],
    "experience": [],
    "education": {
        "level": {"name": "Высшее"},
        "primary": [
            {
                "name": "МГУ",
                "organization": "ВМК",
                "result": "Прикладная математика",
                "year": 2010,
                "education_level": {"name": "Высшее"},
            }
        ],
    },
    "language": [{"name": "Английский", "level": {"name": "C1"}}],
    "professional_roles": [{"name": "DevOps-инженер"}],
    "area": {"name": "Москва"},
    "salary": {"amount": 200000, "currency": "RUR"},
}


def _operation(resume: dict) -> Operation:
    operation = Operation()
    operation._args = SimpleNamespace()
    operation.tool = MagicMock()
    operation._resume_analysis_cache = {}
    operation.__dict__["api_client"] = MagicMock()
    operation.api_client.get.return_value = resume
    return operation


class TestResumeAnalysis:
    def test_empty_sections_are_omitted(self):
        """Пустое «О СЕБЕ» раньше уходило в промпт пустым разделом."""
        operation = _operation(FULL_RESUME)

        text = operation._analyze_resume_heavy({"id": "resume-1"})

        assert "О СЕБЕ" not in text
        assert "ОПЫТ РАБОТЫ" not in text

    def test_education_languages_roles_are_included(self):
        operation = _operation(FULL_RESUME)

        text = operation._analyze_resume_heavy({"id": "resume-1"})

        assert "ОБРАЗОВАНИЕ" in text
        assert "МГУ" in text
        assert "ЯЗЫКИ" in text
        assert "Английский — C1" in text
        assert "ПРОФЕССИОНАЛЬНЫЕ РОЛИ" in text
        assert "DevOps-инженер" in text
        assert "Город: Москва" in text


class TestVacancyContext:
    def test_employer_and_key_skills_are_added(self):
        operation = Operation()
        operation._args = SimpleNamespace()
        operation.tool = MagicMock()

        text = operation._build_vacancy_context(
            {"id": "1", "name": "DevOps"},
            full_vacancy={
                "id": "1",
                "description": "<p>Описание</p>",
                "employer": {"name": "ООО Ромашка"},
                "key_skills": [{"name": "Linux"}, {"name": "Docker"}],
            },
        )

        assert text.startswith("Вакансия: DevOps")
        assert "Работодатель: ООО Ромашка" in text
        assert "Описание: Описание" in text
        assert "Ключевые навыки: Linux, Docker" in text
