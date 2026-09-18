FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/
COPY static/ static/

EXPOSE 8000

# $PORT is injected by most free hosting platforms (Render, Hugging Face
# Spaces, Fly.io...); default to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
