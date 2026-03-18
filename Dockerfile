FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .
RUN pip install --no-cache-dir -e .

# Data dirs
RUN mkdir -p models/local_store models/gguf models/adapters models/versions \
    datasets/raw datasets/processed jobs/queue jobs/logs logs db

EXPOSE 8000

CMD ["llp", "serve"]
