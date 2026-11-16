# --- етап 1: збірка оточення за uv.lock і індексу ---------------------------
FROM python:3.13-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# dependencies first: this layer is cached until uv.lock changes
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

# then the project itself, not editable, so the venv works without src/
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# the index is built at image build time from the committed demo corpus
COPY demo ./demo
RUN .venv/bin/findex index demo/simplewiki-4000.jsonl --out index.pkl --positions


# --- етап 2: лише те, що потрібно для запуску -------------------------------
FROM python:3.13-slim

RUN useradd --create-home --uid 10001 findex

WORKDIR /app
COPY --from=build --chown=findex:findex /app/.venv /app/.venv
COPY --from=build --chown=findex:findex /app/demo /app/demo
COPY --from=build --chown=findex:findex /app/index.pkl /app/index.pkl

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    INDEX_PATH=/app/index.pkl \
    PORT=8000 \
    WEB_CONCURRENCY=1

USER findex
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"

# sh -c so that $PORT from the platform is expanded at start
CMD ["sh", "-c", "exec uvicorn findex.web.app:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY} --proxy-headers"]
