# Tobio backend – deps cached in a separate layer for fast redeploys
FROM python:3.12-slim

# Ultralytics/Matplotlib writable config dirs (avoids Render warnings)
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps first so this layer is cached when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets PORT at runtime; default for local runs
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
