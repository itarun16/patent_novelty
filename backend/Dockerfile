# =====================================
# BASE IMAGE
# =====================================
FROM python:3.11-slim

WORKDIR /app

# =====================================
# SYSTEM DEPENDENCIES
# =====================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# =====================================
# INSTALL PYTHON DEPENDENCIES
# =====================================
COPY requirements.txt .

RUN pip install --upgrade pip

# Install CPU torch first
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# =====================================
# COPY APP
# =====================================
COPY . .

ENV PYTHONUNBUFFERED=1

# Railway PORT support
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]