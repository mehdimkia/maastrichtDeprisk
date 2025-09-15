# Dockerfile (at repo root)
FROM python:3.11-slim

# xgboost runtime needs libgomp
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install API deps
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the **entire** repo (brings in src/, api/, models/, etc.)
COPY . /app

# The FastAPI service lives in /app/api
WORKDIR /app/api
ENV PORT=8000 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
EXPOSE 8000

# Use Render's $PORT if present
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
