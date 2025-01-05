FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

COPY . /project

WORKDIR /project

RUN poetry install --no-root

RUN poetry run pre-commit install