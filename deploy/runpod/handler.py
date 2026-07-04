"""
RunPod Serverless handler for neuromodulated LLM inference (Deploy phase D0).

Wraps the existing in-process generation chain
(``api.model_manager`` -> ``LocalModelInterface.generate_text(pack_name=...)`` ->
``NeuromodTool.apply`` -> logits processors -> ``model.generate``) so a frozen model is
served on a RunPod Serverless worker that **scales to zero** and bills per GPU-second only
while a request is executing.

Design
------
* The heavy model is loaded **once at module scope** (cold start) via :func:`_get_model`, so
  a warm worker reuses it across requests. Neuromod logic is untouched.
* Request parsing (:func:`parse_event`, :func:`messages_to_prompt`) and response shaping
  (:func:`format_response`) are pure functions with no torch/model dependency, so the
  request/response contract is unit-testable on any machine (no GPU needed).
* Torch / model imports are deferred into :func:`_get_model`, and ``runpod`` is imported only
  in :func:`start`, so this module imports cleanly in a minimal environment.

The response shape mirrors the existing ``ChatResponse`` (``api/server.py``) so the Cloudflare
Worker, Streamlit UI, and ``demo/chat.py`` can talk to it with only a URL change.

Streaming (token-by-token) is intentionally out of scope here; it is Deploy phase D1
(a generator handler + SSE proxy).
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_MODEL = os.environ.get("MODEL_NAME", "openai/gpt-oss-20b")

# Repo root (this file is deploy/runpod/handler.py) — server-side jobs shell out from here.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Persist job artifacts on the RunPod network volume so they survive scale-to-zero.
STEERING_DIR = os.environ.get("STEERING_DIR", "/runpod-volume/steering_vectors")
ENDPOINTS_DIR = os.environ.get("ENDPOINTS_DIR", "/runpod-volume/endpoints")

# Optional volume-code overlay. If NEUROMOD_CODE_DIR points at a repo checkout on the network
# volume, prepend it to sys.path so the worker imports neuromod/api from THERE instead of the
# baked /app copy. This lets you ship pure-Python fixes to neuromod.* (e.g. steering) by running
# `git pull` on the volume checkout from a CPU pod — NO image rebuild. Caveats: handler.py itself
# is not overlaid (it's already the running entrypoint from /app), and dependency changes still
# need a rebuild. neuromod.* imports are deferred into functions, so this insert (at import time)
# takes effect before any of them run.
NEUROMOD_CODE_DIR = os.environ.get("NEUROMOD_CODE_DIR")
_CODE_OVERLAY_ACTIVE = bool(NEUROMOD_CODE_DIR and os.path.isdir(NEUROMOD_CODE_DIR))
if _CODE_OVERLAY_ACTIVE:
    sys.path.insert(0, NEUROMOD_CODE_DIR)
    print(f"[overlay] Volume code overlay active: {NEUROMOD_CODE_DIR} (shadows baked /app)", flush=True)
elif NEUROMOD_CODE_DIR:
    print(f"[overlay] NEUROMOD_CODE_DIR={NEUROMOD_CODE_DIR} not a directory; using baked /app", flush=True)

# Model registry (Deploy D2): friendly alias -> HF repo id. These are the switchable
# endpoints -- the snappy default (20b), the hero (120b, 80GB), and Google's Gemma. Only
# vetted ids load: a client-supplied ``model`` is resolved through this allow-list so a
# browser can't make a scale-to-zero GPU pull an arbitrary/giant checkpoint. Set
# ``ALLOW_ANY_MODEL=1`` to lift the allow-list (dev only).
MODEL_REGISTRY = {
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gemma-3-27b": "google/gemma-3-27b-it",
    "gemma-3-12b": "google/gemma-3-12b-it",
    "gpt2": "gpt2",  # ungated test model (matches the study's --tier 1 / test-mode default)
}
_ALLOWED_HF_IDS = set(MODEL_REGISTRY.values()) | {DEFAULT_MODEL}


def resolve_model(name: Optional[str]) -> str:
    """Resolve a client-supplied model alias/id to a vetted HF repo id.

    * empty -> the endpoint default; a known alias -> its HF id; an already-allowed HF id ->
      itself. Anything else falls back to the default (unless ``ALLOW_ANY_MODEL=1``), so an
      untrusted request cannot load an unvetted model onto the GPU.
    """
    if not name:
        return DEFAULT_MODEL
    name = str(name).strip()
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    if name in _ALLOWED_HF_IDS or os.environ.get("ALLOW_ANY_MODEL") == "1":
        return name
    return DEFAULT_MODEL


# --------------------------------------------------------------------------------------
# Pure request / response helpers (no torch, unit-testable)
# --------------------------------------------------------------------------------------


def messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Flatten a chat message list into a prompt.

    Mirrors the conversion in ``api/server.py`` so behavior matches the existing API.
    """
    prompt = ""
    for message in messages or []:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
        elif role == "system":
            prompt += f"{content}\n"
    if not prompt.endswith("Assistant:"):
        prompt += "Assistant:"
    return prompt


def parse_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a RunPod event into generation parameters.

    Accepts either ``{"input": {...}}`` (RunPod convention) or a bare payload dict.
    Supports ``messages`` (chat) or a raw ``prompt``. Missing fields fall back to defaults.
    """
    payload = event.get("input", event) if isinstance(event, dict) else {}
    if not isinstance(payload, dict):
        payload = {}

    if payload.get("prompt"):
        prompt = str(payload["prompt"])
    elif payload.get("messages"):
        prompt = messages_to_prompt(payload["messages"])
    else:
        # No prompt and no messages -> genuinely empty. Keep it empty (don't let
        # messages_to_prompt([]) fabricate a bare "Assistant:") so the empty-input guards in
        # run_inference / run_image actually fire.
        prompt = ""

    def _num(key, default):
        try:
            return type(default)(payload.get(key, default))
        except (TypeError, ValueError):
            return default

    # Intensity is a MULTIPLIER on pack weights, not a 0-1 percentage, so allow overloading a
    # pack past 1.0 ("dial it up until you see an effect"). Bounded by NEUROMOD_MAX_INTENSITY
    # (default 5.0) as a safety rail against a runaway dose crashing the worker.
    _max_intensity = float(os.environ.get("NEUROMOD_MAX_INTENSITY", "5.0"))
    intensity = max(0.0, min(_max_intensity, float(_num("intensity", 0.5))))

    # Chemistry-lab custom pack: a {name, description, effects:[{effect,weight,direction,parameters}]}
    # dict built in the browser. Passed straight to the neuromod tool, which validates it (unknown
    # effect / out-of-range weight -> apply fails -> honest pack_applied=None). Only accept a dict.
    custom_pack = payload.get("custom_pack")
    if not isinstance(custom_pack, dict) or not custom_pack.get("effects"):
        custom_pack = None

    return {
        "prompt": prompt,
        "pack_name": payload.get("pack_name") or None,
        "custom_pack": custom_pack,
        "intensity": intensity,
        "max_tokens": _num("max_tokens", 128),
        "temperature": _num("temperature", 1.0),
        "top_p": _num("top_p", 1.0),
        "model": resolve_model(payload.get("model")),
        # Server-side job routing (default = a normal chat completion).
        "task": (payload.get("task") or "generate"),
        "output_dir": payload.get("output_dir"),
        "layer": payload.get("layer"),
        "only_tests": payload.get("only_tests"),
        # Image generation (task="image"): all optional; None -> the interface's per-model defaults.
        "image_model": payload.get("image_model"),
        "width": payload.get("width"),
        "height": payload.get("height"),
        "steps": payload.get("steps"),
        "guidance_scale": payload.get("guidance_scale"),
        "seed": payload.get("seed"),
        "return_latents": bool(payload.get("return_latents", False)),
    }


def format_response(
    text: str,
    parsed: Dict[str, Any],
    emotions: Optional[Dict[str, Any]] = None,
    generation_time: float = 0.0,
    tokens_generated: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a ``ChatResponse``-compatible dict."""
    if tokens_generated is None:
        tokens_generated = len((text or "").split())
    return {
        "response": text,
        "emotions": emotions or {},
        "pack_applied": parsed.get("pack_name"),
        "intensity": parsed.get("intensity"),
        "tokens_generated": tokens_generated,
        "generation_time": round(generation_time, 4),
        "model_type": "local",
        "model": parsed.get("model"),
    }


def error_response(message: str) -> Dict[str, Any]:
    """RunPod error envelope."""
    return {"error": message}


# --------------------------------------------------------------------------------------
# Cost observability (Deploy D4): per-request GPU-seconds + cold-start accounting
# --------------------------------------------------------------------------------------


def billing_record(parsed: Dict[str, Any], generation_time: float,
                   cold_start: Optional[float] = None) -> Dict[str, Any]:
    """A structured, parseable record of what a request will be billed for.

    ``gpu_seconds`` is the wall-clock the worker held the GPU for this request (the unit
    RunPod Serverless bills on). ``cold_start_seconds`` is set only on the request that paid
    to page a model in (the first request to a cold worker), so it isn't double-counted.
    """
    return {
        "model": parsed.get("model"),
        "pack": parsed.get("pack_name"),
        "gpu_seconds": round(float(generation_time), 4),
        "cold_start_seconds": round(float(cold_start), 4) if cold_start else None,
    }


def log_billing(record: Dict[str, Any]) -> None:
    """Emit the billing record as a single ``[billing] {json}`` line for log scrapers."""
    print(f"[billing] {json.dumps(record)}", flush=True)


# --------------------------------------------------------------------------------------
# Model singleton (heavy — loaded once at cold start)
# --------------------------------------------------------------------------------------

_MODEL = None
_MODEL_NAME: Optional[str] = None
_COLD_START_SECONDS: Optional[float] = None


def _get_model(model_name: str = DEFAULT_MODEL):
    """Load and cache the neuromod-capable model interface once per worker.

    Uses ``LocalModelInterface`` directly (not the legacy ``ModelManager``, whose
    ``generate_text`` signature drops ``pack_name`` — that path errors on every request and
    never applies a pack). ``test_mode=False`` forces the real production load (quantization /
    MXFP4 for gpt-oss), not the small-model test path. Imports torch/neuromod lazily so this
    module stays importable without a GPU. The model-load time is recorded for cold-start
    accounting (Deploy D4).
    """
    global _MODEL, _MODEL_NAME, _COLD_START_SECONDS
    if _MODEL is not None and _MODEL_NAME == model_name:
        return _MODEL

    from api.model_manager import LocalModelInterface  # deferred heavy import

    t0 = time.time()
    iface = LocalModelInterface(model_name, model_type="causal", test_mode=False)
    _COLD_START_SECONDS = time.time() - t0
    _MODEL = iface
    _MODEL_NAME = model_name
    return _MODEL


def _pop_cold_start() -> Optional[float]:
    """Return-and-clear the pending cold-start duration (charged to one request only)."""
    global _COLD_START_SECONDS
    v = _COLD_START_SECONDS
    _COLD_START_SECONDS = None
    return v


def run_inference(parsed: Dict[str, Any], model=None) -> Dict[str, Any]:
    """Execute one neuromodulated generation and shape the response.

    ``model`` may be injected (any object exposing ``generate_text(prompt, max_tokens,
    temperature, top_p, pack_name)``) — used by tests to avoid loading a real model.
    """
    start = time.time()
    if model is None:
        model = _get_model(parsed["model"])
    cold = _pop_cold_start()

    result = model.generate_text(
        prompt=parsed["prompt"],
        max_tokens=parsed["max_tokens"],
        temperature=parsed["temperature"],
        top_p=parsed["top_p"],
        pack_name=parsed["pack_name"],
        custom_pack=parsed.get("custom_pack"),
        intensity=parsed["intensity"],
    )

    if isinstance(result, dict):
        text = result.get("text", "")
        emotions = result.get("emotions", {})
        tokens = result.get("tokens_generated")
        reasoning = result.get("reasoning")
        neuromod_applied = result.get("neuromod_applied", None)
        neuromod_error = result.get("neuromod_error")
    else:
        text, emotions, tokens, reasoning = result, {}, None, None
        neuromod_applied, neuromod_error = None, None

    generation_time = time.time() - start
    response = format_response(
        text, parsed, emotions=emotions,
        generation_time=generation_time, tokens_generated=tokens,
    )
    # A custom (chemistry-lab) pack has no pack_name; label the applied pack by its custom name.
    custom = parsed.get("custom_pack")
    if custom and neuromod_applied is not False:
        response["pack_applied"] = custom.get("name") or "custom"
    # Honesty: only claim pack_applied if it actually applied. A requested-but-failed pack (named
    # OR custom) reports pack_applied=None + neuromod_error so the client isn't told the dose landed.
    if (parsed.get("pack_name") or custom) and neuromod_applied is False:
        response["pack_applied"] = None
        response["neuromod_error"] = neuromod_error or "pack not applied"
    response["neuromod_applied"] = neuromod_applied
    record = billing_record(parsed, generation_time, cold)
    response.update({"gpu_seconds": record["gpu_seconds"],
                     "cold_start_seconds": record["cold_start_seconds"],
                     "reasoning": reasoning})
    log_billing(record)
    return response


# --------------------------------------------------------------------------------------
# Streaming inference (Deploy D1)
# --------------------------------------------------------------------------------------


def chunk_text(text: str) -> List[str]:
    """Split text into word-sized pieces (keeping trailing whitespace) for streaming.

    Pure + testable. Used as the fallback when the model has no native token stream.
    """
    return re.findall(r"\S+\s*", text or "")


def run_inference_stream(parsed: Dict[str, Any], model=None) -> Iterator[Dict[str, Any]]:
    """Yield incremental ``{"chunk": ...}`` events, then a final ``{"done": True, ...}``.

    Prefers a model that natively streams (``generate_text_stream``); otherwise generates
    the full response and re-emits it word-by-word so the client still streams. The final
    event carries the aggregated ``ChatResponse`` payload.
    """
    start = time.time()
    if model is None:
        model = _get_model(parsed["model"])
    cold = _pop_cold_start()

    emotions: Dict[str, Any] = {}
    tokens: Optional[int] = None

    if hasattr(model, "generate_text_stream"):
        pieces: List[str] = []
        for chunk in model.generate_text_stream(
            prompt=parsed["prompt"], max_tokens=parsed["max_tokens"],
            temperature=parsed["temperature"], top_p=parsed["top_p"],
            pack_name=parsed["pack_name"], custom_pack=parsed.get("custom_pack"),
            intensity=parsed["intensity"],
        ):
            pieces.append(chunk)
            yield {"chunk": chunk}
        text = "".join(pieces)
    else:
        result = model.generate_text(
            prompt=parsed["prompt"], max_tokens=parsed["max_tokens"],
            temperature=parsed["temperature"], top_p=parsed["top_p"],
            pack_name=parsed["pack_name"], custom_pack=parsed.get("custom_pack"),
            intensity=parsed["intensity"],
        )
        if isinstance(result, dict):
            text = result.get("text", "")
            emotions = result.get("emotions", {})
            tokens = result.get("tokens_generated")
        else:
            text = result
        for piece in chunk_text(text):
            yield {"chunk": piece}

    generation_time = time.time() - start
    record = billing_record(parsed, generation_time, cold)
    log_billing(record)
    final = format_response(text, parsed, emotions=emotions,
                            generation_time=generation_time, tokens_generated=tokens)
    # Honesty (streaming): reflect whether the pack actually applied (set by generate_text_stream).
    nm = getattr(model, "_last_stream_neuromod", None)
    if nm is not None:
        final["neuromod_applied"] = nm.get("neuromod_applied")
        if parsed.get("pack_name") and nm.get("neuromod_applied") is False:
            final["pack_applied"] = None
            final["neuromod_error"] = nm.get("neuromod_error") or "pack not applied"
    # Harmony (gpt-oss) reasoning: the analysis channel is captured during streaming (only the
    # final channel was streamed as the answer). Surface it so the UI can show an expandable
    # "how it was thinking" panel. None for non-harmony models.
    final["reasoning"] = getattr(model, "_last_stream_reasoning", None)
    final.update({"gpu_seconds": record["gpu_seconds"],
                  "cold_start_seconds": record["cold_start_seconds"]})
    yield {"done": True, **final}


def stream_handler(event: Dict[str, Any], model=None) -> Iterator[Dict[str, Any]]:
    """RunPod generator handler: yields streaming chunks then the final aggregate.

    Non-``generate`` tasks (warmup/steering/endpoints) aren't streaming — they run to
    completion and yield a single result dict, so they work under this generator registration
    (STREAMING=1) exactly as under the sync handler.
    """
    parsed = parse_event(event)
    if parsed["task"] != "generate":
        try:
            yield dispatch_task(parsed, model=model)
        except Exception as exc:  # pragma: no cover - defensive
            yield error_response(f"{type(exc).__name__}: {exc}")
        return
    if not parsed["prompt"]:
        yield error_response("No prompt or messages provided.")
        return
    try:
        for item in run_inference_stream(parsed, model=model):
            yield item
    except Exception as exc:  # pragma: no cover - defensive
        yield error_response(f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------------------
# Server-side jobs (run heavy work ON the warm worker, so the full study stays serverless)
# --------------------------------------------------------------------------------------

TASKS = ("generate", "image", "warmup", "steering", "endpoints", "diag")

# Image (Stable Diffusion) singleton — loaded lazily on the first image request and cached for
# the warm worker, alongside the LLM. Keyed by the resolved SD model id.
_IMAGE_MODEL = None
_IMAGE_MODEL_NAME: Optional[str] = None


def _get_image_model(model_name: Optional[str] = None):
    """Load and cache the neuromodulated SD interface once per worker (per SD model id)."""
    global _IMAGE_MODEL, _IMAGE_MODEL_NAME
    from api.image_model import NeuromodImageInterface, resolve_image_model  # deferred heavy import

    resolved = resolve_image_model(model_name)
    if _IMAGE_MODEL is not None and _IMAGE_MODEL_NAME == resolved:
        return _IMAGE_MODEL
    _IMAGE_MODEL = NeuromodImageInterface(resolved)
    _IMAGE_MODEL_NAME = resolved
    return _IMAGE_MODEL


def run_image(parsed: Dict[str, Any], image_model=None) -> Dict[str, Any]:
    """Generate one neuromodulated image and shape a response (base64 PNG data URL).

    ``image_model`` may be injected by tests (any object exposing ``generate(...)``) to avoid a
    real SD load. The SD model is chosen per endpoint via the ``IMAGE_MODEL`` env; the browser
    can only override within an allow-list (see ``resolve_image_model``).
    """
    if not parsed.get("prompt"):
        return error_response("No prompt provided for image generation.")
    start = time.time()
    cold = None
    if image_model is None:
        t0 = time.time()
        image_model = _get_image_model(parsed.get("image_model"))
        load = time.time() - t0
        cold = load if load > 1.0 else None  # first call pays the SD load; ~0 when warm
    result = image_model.generate(
        prompt=parsed["prompt"], pack_name=parsed.get("pack_name"),
        intensity=parsed.get("intensity", 0.5),
        width=parsed.get("width"), height=parsed.get("height"),
        steps=parsed.get("steps"), guidance_scale=parsed.get("guidance_scale"),
        seed=parsed.get("seed"), return_latents=parsed.get("return_latents", False),
    )
    gpu_seconds = round(time.time() - start, 4)
    result.setdefault("task", "image")
    result["model"] = parsed.get("model")
    result["gpu_seconds"] = gpu_seconds
    result["cold_start_seconds"] = round(cold, 4) if cold else None
    log_billing({"model": _IMAGE_MODEL_NAME or parsed.get("image_model"), "pack": "image",
                 "gpu_seconds": gpu_seconds, "cold_start_seconds": result["cold_start_seconds"]})
    return result


def run_diag(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Report the worker's runtime versions + GPU (no model load) for remote debugging.

    The gpt-oss MXFP4 path is version-sensitive (transformers / kernels / triton must be a
    compatible set); this lets you read exactly what's installed over HTTP on a shell-less
    serverless box. Optionally does a cheap ``AutoConfig`` load to surface config-level errors.
    """
    info: Dict[str, Any] = {"ok": True, "task": "diag", "model": parsed["model"],
                            "steering_dir": STEERING_DIR, "endpoints_dir": ENDPOINTS_DIR}
    try:
        import importlib.metadata as md
        for pkg in ("transformers", "kernels", "triton", "torch", "accelerate", "bitsandbytes"):
            try:
                info[pkg] = md.version(pkg)
            except Exception:
                info[pkg] = "(not installed)"
    except Exception as exc:  # pragma: no cover
        info["metadata_error"] = str(exc)
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda_build"] = torch.version.cuda  # e.g. "12.8" — must be runnable by the host driver
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        info["torch_error"] = str(exc)
    try:
        # Host NVIDIA driver — if torch_cuda_build needs a newer CUDA than this driver supports,
        # cuda_available will be False (the gpt-oss-120b failure mode: cu130 torch on a 12.8 driver).
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        info["driver_version"] = (out.stdout or "").strip().splitlines()[0] if out.stdout else None
    except Exception:
        info["driver_version"] = None
    # Volume presence + free space — the gpt-oss-120b weights need ~63GB on the volume
    # (HF_HOME). A near-full/tiny volume shows up here as low free_gb (the "No space left on
    # device" download failure mode).
    info["volume_mounted"] = os.path.isdir("/runpod-volume")
    import shutil
    hf_home = os.environ.get("HF_HOME", "/root/.cache/huggingface")
    info["hf_home"] = hf_home
    for label, path in (("root", "/"), ("volume", "/runpod-volume"), ("tmp", "/tmp"),
                        ("hf_home", hf_home if os.path.isdir(hf_home) else "/")):
        try:
            u = shutil.disk_usage(path)
            info[f"disk_{label}_free_gb"] = round(u.free / 1e9, 1)
            info[f"disk_{label}_total_gb"] = round(u.total / 1e9, 1)
        except Exception:
            info[f"disk_{label}_free_gb"] = None
    # Which code is the worker actually running? Report the overlay state and the git HEAD of
    # both the baked copy and the overlay so you can confirm a fix is deployed WITHOUT a rebuild.
    info["code_overlay_active"] = _CODE_OVERLAY_ACTIVE
    info["code_overlay_dir"] = NEUROMOD_CODE_DIR

    def _git_head(path):
        try:
            r = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True, timeout=10)
            return (r.stdout or "").strip() or None
        except Exception:
            return None

    info["git_head_baked"] = _git_head(_REPO_ROOT)
    info["git_head_overlay"] = _git_head(NEUROMOD_CODE_DIR) if _CODE_OVERLAY_ACTIVE else None
    # The path Python will actually import neuromod from (proves the overlay is winning).
    try:
        import importlib.util as _ilu
        spec = _ilu.find_spec("neuromod")
        info["neuromod_import_path"] = getattr(spec, "origin", None) if spec else None
    except Exception as exc:
        info["neuromod_import_path"] = f"(error: {exc})"
    return info


def build_job_command(task: str, parsed: Dict[str, Any]) -> "tuple[List[str], str]":
    """Build the argv + output dir for a server-side job. Pure + unit-testable.

    Reuses the existing repo scripts verbatim (no reimplementation): steering-vector
    regeneration and the internal-telemetry endpoint battery run on the worker (which already
    has the GPU), writing to the network volume so results persist across scale-to-zero.
    """
    model = parsed["model"]
    if task == "steering":
        out = parsed.get("output_dir") or STEERING_DIR
        cmd = [sys.executable, "scripts/generate_steering_vectors.py",
               "--model", model, "--output-dir", out]
        if parsed.get("layer") is not None:
            cmd += ["--layer", str(parsed["layer"])]
        return cmd, out
    if task == "endpoints":
        out = parsed.get("output_dir") or ENDPOINTS_DIR
        pack = parsed.get("pack_name") or "lsd"
        cmd = [sys.executable, "scripts/calculate_endpoints.py",
               "--pack", pack, "--model", model, "--output-dir", out, "--skip-completed"]
        if parsed.get("only_tests"):
            cmd += ["--only-tests", *[str(t) for t in parsed["only_tests"]]]
        return cmd, out
    raise ValueError(f"no command for task {task!r}")


def run_job(task: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Run a subprocess job on the worker and report produced artifacts (on the volume)."""
    cmd, out = build_job_command(task, parsed)
    os.makedirs(out, exist_ok=True)
    start = time.time()
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
    produced = sorted(glob.glob(os.path.join(out, "**", "*"), recursive=True))
    result = {
        "ok": proc.returncode == 0,
        "task": task,
        "model": parsed["model"],
        "output_dir": out,
        "returncode": proc.returncode,
        "gpu_seconds": round(time.time() - start, 4),
        "artifacts": [p for p in produced if os.path.isfile(p)][-50:],
        "stdout_tail": (proc.stdout or "")[-1500:],
    }
    if proc.returncode != 0:
        result["error"] = (proc.stderr or proc.stdout or "").strip()[-1500:]
    # Return the endpoints JSON inline so the HTTP caller gets the data (files live on the
    # worker's volume, not the client). Newest matching file for the requested pack.
    if task == "endpoints" and result["ok"]:
        pack = parsed.get("pack_name") or "lsd"
        matches = sorted(glob.glob(os.path.join(out, f"endpoints_{pack}_*.json")))
        if matches:
            try:
                with open(matches[-1]) as fh:
                    result["endpoints_json"] = json.load(fh)
            except Exception as exc:
                result["endpoints_json_error"] = str(exc)
    log_billing({"model": parsed["model"], "pack": task, "gpu_seconds": result["gpu_seconds"],
                 "cold_start_seconds": None})
    return result


def run_warmup(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-load the model on the worker (charges one cold start, then returns).

    Also reports the model's parameter device distribution so you can confirm gpt-oss's MoE
    experts are all on the GPU. Any params on 'cpu'/'meta' are what cause the intermittent
    "grouped_mm mat2 is on cpu" crash at generation.
    """
    start = time.time()
    iface = _get_model(parsed["model"])
    cold = _pop_cold_start()

    device_counts: Dict[str, int] = {}
    hf_device_map = None
    try:
        model = getattr(iface, "model", None)
        if model is not None:
            for p in model.parameters():
                d = str(p.device)
                device_counts[d] = device_counts.get(d, 0) + 1
            hf_device_map = getattr(model, "hf_device_map", None)
            if hf_device_map is not None:
                # summarize which devices the map places layers on (not the full per-layer dump)
                hf_device_map = sorted({str(v) for v in hf_device_map.values()})
    except Exception as exc:  # pragma: no cover - defensive
        device_counts = {"error": str(exc)}

    all_on_gpu = bool(device_counts) and all(
        k.startswith("cuda") for k in device_counts if k != "error")
    return {"ok": True, "task": "warmup", "model": parsed["model"],
            "cold_start_seconds": round(cold, 4) if cold else None,
            "warmup_seconds": round(time.time() - start, 4),
            "param_devices": device_counts,       # e.g. {"cuda:0": 700} — any "cpu"/"meta" is the bug
            "hf_device_map_devices": hf_device_map,
            "all_params_on_gpu": all_on_gpu}


def run_steering_inprocess(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Generate steering vectors on the ALREADY-WARM model (no subprocess reload).

    Reuses the model loaded by :func:`_get_model` and only runs forward passes over the
    prompt-pair dataset, so the whole job is minutes — it fits a serverless execution window.
    (The old subprocess path reloaded ~63GB every time, which is what blew the timeout.)
    Vectors are written to ``STEERING_DIR`` as ``<type>_layer<layer>.pt`` — the exact name the
    pack loader looks for — so inference picks them up via ``STEERING_DIR``.
    """
    import json as _json
    import torch
    from neuromod.steering_generator import SteeringVectorGenerator
    from neuromod.effects import steering_model_slug

    start = time.time()
    iface = _get_model(parsed["model"])
    cold = _pop_cold_start()
    # Nest under a per-model subdir so vectors for different served models coexist on the
    # one network volume; the loader (resolve_steering_vector_path) finds them via MODEL_NAME.
    out = parsed.get("output_dir")
    if not out:
        out = STEERING_DIR
        slug = steering_model_slug(parsed["model"])
        if slug and os.path.basename(os.path.normpath(out)) != slug:
            out = os.path.join(out, slug)
    os.makedirs(out, exist_ok=True)
    layer = parsed.get("layer")
    layer = -1 if layer is None else int(layer)  # match the pack loader (<type>_layer-1.pt)
    min_pairs = int(parsed.get("min_pairs") or 100)
    validate = bool(parsed.get("validate", False))  # off by default so a new arch still emits

    dataset = os.path.join(_REPO_ROOT, "datasets", "steering_prompts.jsonl")
    types = set()
    with open(dataset) as fh:
        for line in fh:
            try:
                d = _json.loads(line)
                if d.get("steering_type"):
                    types.add(d["steering_type"])
            except Exception:
                continue
    only = parsed.get("steering_type")
    type_list = [only] if only else sorted(types)

    gen = SteeringVectorGenerator(iface.model, iface.tokenizer)
    made: List[str] = []
    failed: Dict[str, str] = {}
    for st in type_list:
        try:
            vec = gen.compute_vector_robust(dataset_path=dataset, steering_type=st,
                                            layer_idx=layer, use_pca=True, validate=validate,
                                            min_pairs=min_pairs)
            path = os.path.join(out, f"{st}_layer{layer}.pt")
            torch.save(vec, path)
            made.append(os.path.basename(path))
        except Exception as exc:  # per-type failure shouldn't sink the whole job
            failed[st] = str(exc)[-300:]

    result = {"ok": len(made) > 0, "task": "steering", "model": parsed["model"],
              "output_dir": out, "generated": made, "failed": failed,
              "gpu_seconds": round(time.time() - start, 2),
              "cold_start_seconds": round(cold, 2) if cold else None}
    log_billing({"model": parsed["model"], "pack": "steering",
                 "gpu_seconds": result["gpu_seconds"], "cold_start_seconds": result["cold_start_seconds"]})
    return result


def dispatch_task(parsed: Dict[str, Any], model=None) -> Dict[str, Any]:
    """Route a non-``generate`` task to its handler. Returns a result dict."""
    task = parsed["task"]
    if task == "diag":
        return run_diag(parsed)
    if task == "image":
        return run_image(parsed, image_model=model)
    if task == "warmup":
        return run_warmup(parsed)
    if task == "steering":
        return run_steering_inprocess(parsed)  # reuse warm model — fits the serverless window
    if task == "endpoints":
        return run_job(task, parsed)
    return error_response(f"unknown task: {task!r} (expected one of {TASKS})")


# --------------------------------------------------------------------------------------
# RunPod entrypoints
# --------------------------------------------------------------------------------------


def handler(event: Dict[str, Any], model=None) -> Dict[str, Any]:
    """RunPod Serverless sync handler: event -> response dict."""
    try:
        parsed = parse_event(event)
        if parsed["task"] != "generate":
            return dispatch_task(parsed, model=model)
        if not parsed["prompt"]:
            return error_response("No prompt or messages provided.")
        return run_inference(parsed, model=model)
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        return error_response(f"{type(exc).__name__}: {exc}")


def start():  # pragma: no cover - requires the runpod runtime
    """Start the RunPod Serverless worker. Called as the container entrypoint."""
    import runpod  # deferred; only present in the RunPod image

    # Optionally warm the model at boot so the first request isn't the cold-start victim.
    if os.environ.get("WARM_ON_START", "0") == "1":
        _get_model(DEFAULT_MODEL)

    # STREAMING=1 (default) registers the generator handler; `return_aggregate_stream`
    # keeps /runsync returning the aggregated result for non-streaming callers.
    if os.environ.get("STREAMING", "1") == "1":
        runpod.serverless.start({"handler": stream_handler, "return_aggregate_stream": True})
    else:
        runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    start()
