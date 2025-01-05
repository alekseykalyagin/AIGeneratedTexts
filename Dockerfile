# Используем базовый образ Python 3.10
FROM python:3.10-slim

# Обновление и установка необходимых системных пакетов
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Установка Poetry
RUN pip install --no-cache-dir poetry

# Копируем конфиг pre-commit
COPY .pre-commit-config.yaml ./

# Копируем файлы Poetry для установки зависимостей
COPY pyproject.toml poetry.lock ./

# Установка зависимостей проекта через Poetry
RUN poetry install --no-root --no-interaction --no-ansi

# Копируем весь проект
COPY . /workspace

# Устанавливаем рабочую директорию
WORKDIR /workspace

# Настройка DVC для локального хранилища
RUN dvc init
RUN mkdir dvc_storage
RUN dvc remote add -d local_storage /workspace/dvc_storage
RUN dvc config core.remote local_storage

# Установка pre-commit
RUN pip install --no-cache-dir pre-commit && pre-commit install
