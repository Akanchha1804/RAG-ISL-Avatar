# 3.12: numpy 2.5.4 (see requirements.txt) requires >=3.12, so a 3.11 base
# image fails the install step outright.
FROM python:3.12-slim

WORKDIR /app

# CPU torch and faiss link against the OpenMP runtime, which the slim base
# does not ship; without it `import torch`/`faiss` fails at container start
# (the image still builds, which is why CI smoke-starts the image).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first (much smaller than CUDA wheels); keep index for resolution
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY backend/ .

# The VRM avatar lives outside backend/, so the repo layout paths.py resolves
# first is absent here; carry a copy at the backend-local fallback path or
# /avatar/model.vrm 404s and the web viewport renders no avatar.
COPY ["ISL-3D Avatar/Assets/model_isl.vrm", "./avatar/model_isl.vrm"]

# paths.py: DATA_DIR/ISL_MediaPipe/... primary; BACKEND_DIR/* fallback when DATA_DIR unset
# Compose sets DATA_DIR=/data and mounts landmarks/mapping there.
ENV DATA_DIR=/data \
    PYTHONUNBUFFERED=1 \
    ISL_WARMUP_MODELS=1

RUN mkdir -p /data/ISL_MediaPipe/output_landmarks \
    landmarks index models/isl_gloss_t5

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
