# RunPod Serverless image for neuromodulated LLM inference (Deploy D0).
#
# Scale-to-zero economics: this image runs as a RunPod Serverless worker (min_workers=0),
# billed per GPU-second only while handler() executes. Model weights should live on a RunPod
# **network volume** (mounted at /runpod-volume) so cold starts don't re-download them.

# Base: any RunPod PyTorch image on **CUDA 12.8** with Python 3.11. The exact torch in the base
# doesn't matter — we force a driver-matched torch below — but keep CUDA 12.8 system libs.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Cache HF weights on the mounted network volume so cold starts are fast.
    HF_HOME=/runpod-volume/hf \
    # Disable the hf_xet backend: its concurrent "background writer" is flaky on NETWORK
    # filesystems (the RunPod network volume) — fails with "Background writer channel closed".
    # Classic sequential HTTP downloads write reliably to the volume.
    HF_HUB_DISABLE_XET=1 \
    MODEL_NAME=openai/gpt-oss-20b \
    STEERING_DIR=/runpod-volume/steering_vectors \
    ENDPOINTS_DIR=/runpod-volume/endpoints \
    # Volume code overlay: if a repo checkout exists here on the network volume, the handler
    # imports neuromod/api from it instead of the baked /app copy — so pure-Python fixes to
    # neuromod.* ship via `git pull` on the volume (from a CPU pod), NO rebuild. No-op until the
    # dir exists (handler guards on isdir), so this default is safe.
    NEUROMOD_CODE_DIR=/runpod-volume/code/neuromod-llm-poc \
    WARM_ON_START=0

WORKDIR /app

# Install serving deps first (better layer caching).
COPY deploy/runpod/requirements.txt /app/deploy/runpod/requirements.txt
RUN pip install --no-cache-dir -r /app/deploy/runpod/requirements.txt

# Copy the repo (neuromod package, api/model_manager chain, packs, handler).
COPY . /app
RUN pip install --no-cache-dir -e .

# CRITICAL — pin torch to a CUDA build the RunPod host driver can run.
# Observed on the H200 hosts: driver 570.195.03, which supports up to **CUDA 12.8**. Left alone,
# `pip install -e .` / transformers re-resolve torch to the latest wheel (e.g. 2.12+cu130), and a
# **CUDA-13 runtime on a 12.8 driver makes torch.cuda.is_available() == False** — the model then
# falls to a doomed CPU dequant and dies with "Either a revision or a version must be specified".
# Forcing the cu128 build here, as the LAST pip step, guarantees a driver-compatible torch that
# still ships Triton >= 3.4 for gpt-oss MXFP4. (If your host driver is >= 580, cu130 is fine and
# you can drop this; check the `diag` task's driver line.)
RUN pip install --no-cache-dir --force-reinstall \
      torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu128

# Fail the build (free) — not a paid cold start — on either known gpt-oss load blocker:
#   (1) torch not a CUDA 12.x build (host driver supports <= 12.8), or
#   (2) kernels >= 0.15 (breaks the MXFP4 Hub-kernel load — transformers#46291).
RUN python -c "import torch,sys; cu=torch.version.cuda or ''; print('torch',torch.__version__,'cuda',cu); \
sys.exit(0 if cu.startswith('12') else 'ERROR: torch CUDA %r is not 12.x; host driver supports <=12.8'%cu)"
RUN python -c "import importlib.metadata as m; from packaging.version import Version; \
v=Version(m.version('kernels')); print('kernels',v); \
assert v < Version('0.15'), 'kernels>=0.15 breaks gpt-oss MXFP4 (transformers#46291); pin <0.15'"

# RunPod invokes the handler via the serverless runtime started in handler.start().
CMD ["python", "-u", "deploy/runpod/handler.py"]
