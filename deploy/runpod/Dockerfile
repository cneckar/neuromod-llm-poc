# RunPod Serverless image for neuromodulated LLM inference (Deploy D0).
#
# Scale-to-zero economics: this image runs as a RunPod Serverless worker (min_workers=0),
# billed per GPU-second only while handler() executes. Model weights should live on a RunPod
# **network volume** (mounted at /runpod-volume) so cold starts don't re-download them.

# CUDA + PyTorch base. gpt-oss ships MXFP4 weights whose Triton kernels need **Triton >= 3.4
# (PyTorch >= 2.7) on a Hopper GPU (H100/H200)**. torch 2.4 (Triton 3.0) is too old and fails
# with "Either a revision or a version must be specified" when loading the MXFP4 kernels.
# Use the newest runpod/pytorch tag with torch >= 2.7; adjust the tag to one that exists.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Cache HF weights on the mounted network volume so cold starts are fast.
    HF_HOME=/runpod-volume/hf \
    MODEL_NAME=openai/gpt-oss-20b \
    STEERING_DIR=/runpod-volume/steering_vectors \
    ENDPOINTS_DIR=/runpod-volume/endpoints \
    WARM_ON_START=0

WORKDIR /app

# CRITICAL: freeze the base image's CUDA torch stack so nothing downgrades or replaces it with a
# CPU-only / mismatched wheel. transformers>=4.55 would otherwise force a torch re-resolve that
# can leave `torch.cuda.is_available() == False` even on a healthy GPU. PIP_CONSTRAINT is honored
# by every subsequent `pip install`, so torch/torchvision/torchaudio/triton stay exactly as shipped.
RUN pip freeze | grep -Ei '^(torch|torchvision|torchaudio|triton|pytorch-triton)==' \
      > /etc/torch-constraints.txt || true
ENV PIP_CONSTRAINT=/etc/torch-constraints.txt

# Install serving deps first (better layer caching). torch is pinned by the constraint above.
COPY deploy/runpod/requirements.txt /app/deploy/runpod/requirements.txt
RUN pip install --no-cache-dir -r /app/deploy/runpod/requirements.txt

# Copy the repo (neuromod package, api/model_manager chain, packs, handler).
COPY . /app
RUN pip install --no-cache-dir -e .

# Fail the build early if the CUDA torch got clobbered anyway (surfaces here, not at $/cold-start).
RUN python -c "import torch; assert torch.version.cuda, 'CPU-only torch got installed'; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# RunPod invokes the handler via the serverless runtime started in handler.start().
CMD ["python", "-u", "deploy/runpod/handler.py"]
