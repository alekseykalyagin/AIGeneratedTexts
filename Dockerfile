FROM python:3.10-slim

# Установка Poetry
RUN pip install poetry

COPY .pre-commit-config.yaml ./

# Установка зависимостей
COPY pyproject.toml ./
RUN poetry install --no-root

# Копируем исходный код
COPY model/ ./model/
COPY data/ ./data/


# Копируем весь исходный код проекта
COPY . /workspace

# Устанавливаем рабочую директорию
WORKDIR /workspace
