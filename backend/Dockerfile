# Use a slightly larger base with more system libraries
FROM python:3.11-bullseye

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install CPU-only PyTorch wheel + other heavy dependencies separately
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    open-clip-torch==2.0.2 \
    sentence-transformers \
    Pillow \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure Python output is unbuffered
ENV PYTHONUNBUFFERED=1

# Run FastAPI / Uvicorn
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]