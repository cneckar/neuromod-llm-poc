# RunPod Serverless image for neuromodulated LLM inference (Deploy D0).
#
# Scale-to-zero economics: this image runs as a RunPod Serverless worker (min_workers=0),
# billed per GPU-second only while handler() executes. Model weights should live on a RunPod
# **network volume** (mounted at /runpod-volume) so cold starts don't re-download them.

# CUDA + PyTorch base. Pin to a version compatible with the target model's kernels.
# gpt-oss (MXFP4) needs a recent transformers + triton kernels; adjust as needed.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Cache HF weights on the mounted network volume so cold starts are fast.
    HF_HOME=/runpod-volume/hf \
    MODEL_NAME=openai/gpt-oss-20b \
    WARM_ON_START=0

WORKDIR /app

# Install serving deps first (better layer caching).
COPY deploy/runpod/requirements.txt /app/deploy/runpod/requirements.txt
RUN pip install --no-cache-dir -r /app/deploy/runpod/requirements.txt

# Copy the repo (neuromod package, api/model_manager chain, packs, handler).
COPY . /app
RUN pip install --no-cache-dir -e .

# RunPod invokes the handler via the serverless runtime started in handler.start().
CMD ["python", "-u", "deploy/runpod/handler.py"]
