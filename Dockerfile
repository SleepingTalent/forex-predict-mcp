# --------------------------------------------------------------------------- #
# Builder — install uv and sync all production dependencies                   #
# --------------------------------------------------------------------------- #
FROM python:3.13-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
RUN uv sync --frozen --no-dev

# --------------------------------------------------------------------------- #
# Runtime — minimal image, non-root user                                      #
# --------------------------------------------------------------------------- #
FROM python:3.13-slim AS runtime

WORKDIR /app

RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --no-create-home appuser

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser

CMD ["python", "-m", "forex_predict_mcp.server"]
