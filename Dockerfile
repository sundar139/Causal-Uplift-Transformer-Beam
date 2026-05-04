FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY artifacts/reports ./artifacts/reports
COPY artifacts/data ./artifacts/data
COPY models/production ./models/production

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "causal_uplift.serve:app", "--host", "0.0.0.0", "--port", "8080"]
