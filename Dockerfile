# syntax=docker/dockerfile:1
# Multi-stage: build the venv with uv, then a slim non-root runtime.

FROM python:3.12-slim AS build
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 UV_LINK_MODE=copy
RUN pip install --no-cache-dir uv==0.7.19
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"
RUN useradd --create-home --uid 10001 council
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY src ./src
COPY config ./config
COPY migrations ./migrations
COPY alembic.ini ./
USER council
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"
CMD ["uvicorn", "ai_council.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
