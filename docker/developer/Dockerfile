FROM python:3.10.6 as python-base

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV PIP_DEFAULT_TIMEOUT=100
ENV POETRY_VERSION=1.3.2
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VIRTUALENVS_IN_PROJECT=false
ENV POETRY_NO_INTERACTION=1
ENV PYSETUP_PATH="/app"
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"
ENV PYTHONPATH=${PYTHONPATH}:${PWD}

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    curl \
    build-essential

RUN curl -sSL https://install.python-poetry.org | python3 -

COPY pyproject.toml ./

RUN poetry export --with dev -f requirements.txt --output requirements.txt

RUN pip install -r  ./requirements.txt


COPY ./ .

CMD ["python", "pantanalise/main.py"]