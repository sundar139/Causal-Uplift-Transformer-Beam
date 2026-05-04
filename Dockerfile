FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements-serving.txt ./
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src ./src
COPY models/production ./models/production

EXPOSE 8080

CMD ["sh", "-c", "python -m uvicorn causal_uplift.serve:app --host 0.0.0.0 --port ${PORT:-8080}"]
