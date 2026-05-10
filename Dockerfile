# Churn proxy pipeline — ECS/Fargate ready (use task role for S3).
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md pipeline.py ./
COPY config ./config/
COPY src ./src/
COPY tests ./tests/

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
CMD ["python", "pipeline.py", "--config", "config/churn_pipeline.yaml"]
