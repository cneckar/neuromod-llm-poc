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

import json
import os
import re
import time
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_MODEL = os.environ.get("MODEL_NAME", "openai/gpt-oss-20b")

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
    else:
        prompt = messages_to_prompt(payload.get("messages", []))

    def _num(key, default):
        try:
            return type(default)(payload.get(key, default))
        except (TypeError, ValueError):
            return default

    intensity = _num("intensity", 0.5)
    intensity = max(0.0, min(1.0, float(intensity)))

    return {
        "prompt": prompt,
        "pack_name": payload.get("pack_name") or None,
        "intensity": intensity,
        "max_tokens": _num("max_tokens", 128),
        "temperature": _num("temperature", 1.0),
        "top_p": _num("top_p", 1.0),
        "model": resolve_model(payload.get("model")),
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
    """Load and cache the model + neuromod interface once per worker.

    Imports torch / diffusers / neuromod lazily so this module stays importable without a
    GPU. Reuses ``api.model_manager.ModelManager``, which attaches the neuromod hooks. The
    time paid to page a (new) model in is recorded for cold-start accounting (Deploy D4).
    """
    global _MODEL, _MODEL_NAME, _COLD_START_SECONDS
    if _MODEL is not None and _MODEL_NAME == model_name:
        return _MODEL

    from api.model_manager import ModelManager  # deferred heavy import

    t0 = time.time()
    manager = ModelManager()
    manager.load_model(model_name)
    _COLD_START_SECONDS = time.time() - t0
    _MODEL = manager
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
    )

    if isinstance(result, dict):
        text = result.get("text", "")
        emotions = result.get("emotions", {})
        tokens = result.get("tokens_generated")
    else:
        text, emotions, tokens = result, {}, None

    generation_time = time.time() - start
    response = format_response(
        text, parsed, emotions=emotions,
        generation_time=generation_time, tokens_generated=tokens,
    )
    record = billing_record(parsed, generation_time, cold)
    response.update({"gpu_seconds": record["gpu_seconds"],
                     "cold_start_seconds": record["cold_start_seconds"]})
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
            pack_name=parsed["pack_name"],
        ):
            pieces.append(chunk)
            yield {"chunk": chunk}
        text = "".join(pieces)
    else:
        result = model.generate_text(
            prompt=parsed["prompt"], max_tokens=parsed["max_tokens"],
            temperature=parsed["temperature"], top_p=parsed["top_p"],
            pack_name=parsed["pack_name"],
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
    final.update({"gpu_seconds": record["gpu_seconds"],
                  "cold_start_seconds": record["cold_start_seconds"]})
    yield {"done": True, **final}


def stream_handler(event: Dict[str, Any], model=None) -> Iterator[Dict[str, Any]]:
    """RunPod generator handler: yields streaming chunks then the final aggregate."""
    parsed = parse_event(event)
    if not parsed["prompt"]:
        yield error_response("No prompt or messages provided.")
        return
    try:
        for item in run_inference_stream(parsed, model=model):
            yield item
    except Exception as exc:  # pragma: no cover - defensive
        yield error_response(f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------------------
# RunPod entrypoints
# --------------------------------------------------------------------------------------


def handler(event: Dict[str, Any], model=None) -> Dict[str, Any]:
    """RunPod Serverless sync handler: event -> response dict."""
    try:
        parsed = parse_event(event)
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
