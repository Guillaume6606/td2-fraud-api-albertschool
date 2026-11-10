FROM python:3.13-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-package nvidia-nccl-cu13

FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --no-create-home app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY app.py features.py scoring.py settings.py ./
COPY models/model.ubj models/features.json ./models/
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data WEB_CONCURRENCY=1 XGB_NTHREAD=1 PORT=8000
USER app
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
