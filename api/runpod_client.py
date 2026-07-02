"""
Torch-free HTTP client for a deployed RunPod Serverless neuromod endpoint.

Kept free of torch / the heavy model stack so it can be used from a laptop (or CI) to drive
the endpoint without loading any model — you pay for GPU-seconds only while the worker runs.
Two ways to use it:

* :meth:`RunPodModelInterface.generate_text` — the ``generate_text(pack_name=..., intensity=..)``
  contract, for a behavioral sweep (text in, text out).
* :meth:`RunPodModelInterface.run_task` — invoke a server-side *job* (``warmup`` / ``steering`` /
  ``endpoints``) so heavier work (steering-vector regen, the internal-telemetry battery) runs on
  the warm worker and its results land on the network volume — all still scale-to-zero.

See ``deploy/runpod/handler.py`` for the endpoint side and ``scripts/run_remote_study.py`` for a
driver.
"""

from typing import Any, Dict, List, Optional

import requests

RUNPOD_BASE = "https://api.runpod.ai/v2"


class RunPodModelInterface:
    """HTTP client for the RunPod Serverless neuromod endpoint."""

    def __init__(self, endpoint_id: str, api_key: str, model: Optional[str] = None,
                 base: str = RUNPOD_BASE, timeout: int = 3600):
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.model = model
        self.base = base.rstrip("/")
        self.timeout = timeout

    # ---- low-level ---------------------------------------------------------------------

    def _runsync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base}/{self.endpoint_id}/runsync"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = requests.post(url, headers=headers, json={"input": payload}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _extract_output(body: Dict[str, Any]) -> Dict[str, Any]:
        """Pull the final result dict out of a /runsync body.

        A streaming (generator) endpoint with ``return_aggregate_stream=True`` returns
        ``output`` as a list of yielded items; a plain handler returns a dict. Handle both.
        """
        out = body.get("output", body)
        if isinstance(out, list):
            for item in reversed(out):
                if isinstance(item, dict) and (item.get("done") or "response" in item or "ok" in item):
                    return item
            return out[-1] if out and isinstance(out[-1], dict) else {}
        return out if isinstance(out, dict) else {}

    # ---- high-level --------------------------------------------------------------------

    def run_task(self, task: str, **payload) -> Dict[str, Any]:
        """Invoke a server-side job (``warmup`` / ``steering`` / ``endpoints`` / ``generate``)."""
        body = {"task": task, **payload}
        if self.model and "model" not in body:
            body["model"] = self.model
        out = self._extract_output(self._runsync(body))
        if isinstance(out, dict) and out.get("error") and "response" not in out:
            raise RuntimeError(f"RunPod task '{task}' error: {out['error']}")
        return out

    def generate_text(self, prompt: str, max_tokens: int = 100,
                      temperature: float = 1.0, top_p: float = 1.0,
                      pack_name: Optional[str] = None, intensity: float = 0.5,
                      **_ignored) -> Dict[str, Any]:
        """One neuromodulated completion. Returns ``{text, emotions, gpu_seconds, raw}``."""
        payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
                   "top_p": top_p, "pack_name": pack_name, "intensity": intensity}
        if self.model:
            payload["model"] = self.model
        out = self._extract_output(self._runsync(payload))
        if out.get("error") and "response" not in out:
            raise RuntimeError(f"RunPod endpoint error: {out['error']}")
        return {"text": out.get("response", ""), "emotions": out.get("emotions", {}),
                "gpu_seconds": out.get("gpu_seconds"), "raw": out}

    def warmup(self) -> Dict[str, Any]:
        """Pre-load the model on a worker (pays one cold start; useful before a sweep)."""
        return self.run_task("warmup")

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base}/{self.endpoint_id}/health",
                                headers={"Authorization": f"Bearer {self.api_key}"}, timeout=30)
            return resp.status_code == 200
        except Exception:
            return False


def interface_from_env(model: Optional[str] = None) -> "RunPodModelInterface":
    """Build a client from ``RUNPOD_ENDPOINT_ID`` + ``RUNPOD_API_KEY`` env vars."""
    import os
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not endpoint_id or not api_key:
        raise RuntimeError("Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY (never hard-code the key).")
    return RunPodModelInterface(endpoint_id, api_key, model=model)
