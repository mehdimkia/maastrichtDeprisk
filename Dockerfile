# Dockerfile at repo root
FROM python:3.11-slim

# xgboost needs libgomp
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps live in api/requirements.txt
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the whole repo (brings in api/, models/, src/, etc.)
COPY . /app

# run the FastAPI app from the api folder
WORKDIR /app/api
ENV PORT=8000 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
