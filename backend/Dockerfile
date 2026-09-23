FROM python:3.11-slim

WORKDIR /app

# CPU-only torch first (much smaller than CUDA wheels); keep index for resolution
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY backend/ .

# paths.py: DATA_DIR/ISL_MediaPipe/... primary; BACKEND_DIR/* fallback when DATA_DIR unset
# Compose sets DATA_DIR=/data and mounts landmarks/mapping there.
ENV DATA_DIR=/data \
    PYTHONUNBUFFERED=1 \
    ISL_WARMUP_MODELS=1

RUN mkdir -p /data/ISL_MediaPipe/output_landmarks \
    landmarks index models/isl_gloss_t5

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
