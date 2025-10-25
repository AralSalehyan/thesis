# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git ca-certificates netcat-openbsd && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install requirements first for layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Default command (override in compose)
CMD ["python", "-m", "experiments.run_experiment", "--help"]
