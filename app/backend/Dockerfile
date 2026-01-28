# Tobio backend – deps cached in a separate layer for fast redeploys
FROM python:3.10-slim

# cv2 (OpenCV) needs these .so files even with opencv-python-headless on slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 libxcb-shm0 libxcb-xfixes0 libxrender1 libxext6 libsm6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Ultralytics/Matplotlib writable config dirs (avoids Render warnings)
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps first so this layer is cached when only code changes.
# Ultralytics pulls opencv-python (needs libxcb/GUI). Replace with headless for slim/Docker.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python 2>/dev/null || true && \
    pip install --no-cache-dir opencv-python-headless

COPY . .

# Render sets PORT at runtime; default for local runs
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
