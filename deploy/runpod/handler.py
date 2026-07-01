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

import os
import time
from typing import Any, Dict, List, Optional

DEFAULT_MODEL = os.environ.get("MODEL_NAME", "openai/gpt-oss-20b")


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
        "model": payload.get("model") or DEFAULT_MODEL,
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
# Model singleton (heavy — loaded once at cold start)
# --------------------------------------------------------------------------------------

_MODEL = None
_MODEL_NAME: Optional[str] = None


def _get_model(model_name: str = DEFAULT_MODEL):
    """Load and cache the model + neuromod interface once per worker.

    Imports torch / diffusers / neuromod lazily so this module stays importable without a
    GPU. Reuses ``api.model_manager.ModelManager``, which attaches the neuromod hooks.
    """
    global _MODEL, _MODEL_NAME
    if _MODEL is not None and _MODEL_NAME == model_name:
        return _MODEL

    from api.model_manager import ModelManager  # deferred heavy import

    manager = ModelManager()
    manager.load_model(model_name)
    _MODEL = manager
    _MODEL_NAME = model_name
    return _MODEL


def run_inference(parsed: Dict[str, Any], model=None) -> Dict[str, Any]:
    """Execute one neuromodulated generation and shape the response.

    ``model`` may be injected (any object exposing ``generate_text(prompt, max_tokens,
    temperature, top_p, pack_name)``) — used by tests to avoid loading a real model.
    """
    start = time.time()
    if model is None:
        model = _get_model(parsed["model"])

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

    return format_response(
        text, parsed, emotions=emotions,
        generation_time=time.time() - start, tokens_generated=tokens,
    )


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

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    start()
