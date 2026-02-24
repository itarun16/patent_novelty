# Stage 1: Builder (install deps, heavy stuff)
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps minimally
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime (copy only needed artifacts)
FROM python:3.11-slim

WORKDIR /app

# Copy venv from builder (or installed packages)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy your app code + index files
COPY backend/ .

# Optional: Clean up any temp files if needed
RUN rm -rf /root/.cache/pip/* /tmp/*

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]