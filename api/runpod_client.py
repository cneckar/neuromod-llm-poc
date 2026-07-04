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

import time
from typing import Any, Dict, List, Optional

import requests

RUNPOD_BASE = "https://api.runpod.ai/v2"

# Terminal RunPod job states for the async /run + /status flow.
_DONE_STATES = {"COMPLETED"}
_FAIL_STATES = {"FAILED", "CANCELLED", "TIMED_OUT"}


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

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _runsync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base}/{self.endpoint_id}/runsync"
        resp = requests.post(url, headers=self._headers(), json={"input": payload}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _run_async(self, payload: Dict[str, Any], poll_interval: float = 5.0,
                   on_status=None) -> Dict[str, Any]:
        """Submit via /run and poll /status until terminal. For long jobs (steering / the
        endpoint battery on a big model) that would blow the synchronous /runsync window.

        Returns the full status body (with ``output``). Raises on FAILED/CANCELLED/TIMED_OUT
        or if the job doesn't finish within ``self.timeout`` seconds.
        """
        run_url = f"{self.base}/{self.endpoint_id}/run"
        resp = requests.post(run_url, headers=self._headers(), json={"input": payload}, timeout=120)
        resp.raise_for_status()
        job_id = resp.json().get("id")
        if not job_id:
            raise RuntimeError(f"RunPod /run returned no job id: {resp.text[:300]}")

        status_url = f"{self.base}/{self.endpoint_id}/status/{job_id}"
        deadline = time.time() + self.timeout
        last = None
        while time.time() < deadline:
            s = requests.get(status_url, headers=self._headers(), timeout=60)
            s.raise_for_status()
            body = s.json()
            status = body.get("status")
            if status != last:
                if on_status:
                    on_status(status, job_id)
                last = status
            if status in _DONE_STATES:
                return body
            if status in _FAIL_STATES:
                raise RuntimeError(f"RunPod job {job_id} {status}: "
                                   f"{str(body.get('error') or body.get('output'))[:500]}")
            time.sleep(poll_interval)
        raise TimeoutError(f"RunPod job {job_id} did not finish within {self.timeout}s "
                           f"(last status: {last}). Raise the client timeout or the endpoint's "
                           f"Execution Timeout.")

    @staticmethod
    def _extract_output(body: Dict[str, Any]) -> Dict[str, Any]:
        """Pull the final result dict out of a /runsync body.

        A streaming (generator) endpoint with ``return_aggregate_stream=True`` returns
        ``output`` as a list of yielded items; a plain handler returns a dict. Handle both.
        """
        out = body.get("output", body)
        if isinstance(out, list):
            for item in reversed(out):
                if isinstance(item, dict) and (item.get("done") or "response" in item
                                               or "ok" in item or "image" in item):
                    return item
            return out[-1] if out and isinstance(out[-1], dict) else {}
        return out if isinstance(out, dict) else {}

    # ---- high-level --------------------------------------------------------------------

    def run_task(self, task: str, wait: bool = True, poll_interval: float = 5.0,
                 on_status=None, **payload) -> Dict[str, Any]:
        """Invoke a server-side job (``warmup`` / ``steering`` / ``endpoints`` / ``generate``).

        Server-side jobs are long (steering-vector regen and the Table-1 battery on a big model
        run for minutes), so by default this uses the async ``/run`` + ``/status`` flow and polls
        to completion — the synchronous ``/runsync`` window would time out. Pass ``wait=False`` to
        use ``/runsync`` (only for short jobs like ``warmup`` on an already-warm worker).
        """
        body = {"task": task, **payload}
        if self.model and "model" not in body:
            body["model"] = self.model
        raw = self._run_async(body, poll_interval=poll_interval, on_status=on_status) if wait \
            else self._runsync(body)
        out = self._extract_output(raw)
        if isinstance(out, dict) and out.get("error") and "response" not in out:
            raise RuntimeError(f"RunPod task '{task}' error: {out['error']}")
        return out

    def generate_text(self, prompt: str, max_tokens: int = 100,
                      temperature: float = 1.0, top_p: float = 1.0,
                      pack_name: Optional[str] = None, intensity: float = 0.5,
                      wait: bool = True, poll_interval: float = 3.0, on_status=None,
                      **_ignored) -> Dict[str, Any]:
        """One neuromodulated completion. Returns ``{text, emotions, reasoning, gpu_seconds, raw}``.

        Big reasoning models (gpt-oss-120b) — especially on a cold start — routinely take longer
        than RunPod's ~90s ``/runsync`` window, which then returns IN_PROGRESS with NO output (an
        empty reply). So by default this uses the async ``/run`` + poll ``/status`` flow and waits
        for completion. Pass ``wait=False`` for the low-latency ``/runsync`` path on small/warm
        models.
        """
        payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
                   "top_p": top_p, "pack_name": pack_name, "intensity": intensity}
        if self.model:
            payload["model"] = self.model
        raw = self._run_async(payload, poll_interval=poll_interval, on_status=on_status) if wait \
            else self._runsync(payload)
        out = self._extract_output(raw)
        if out.get("error") and "response" not in out:
            raise RuntimeError(f"RunPod endpoint error: {out['error']}")
        return {"text": out.get("response", ""), "emotions": out.get("emotions", {}),
                "reasoning": out.get("reasoning"),
                "gpu_seconds": out.get("gpu_seconds"), "raw": out}

    def generate_image(self, prompt: str, pack_name: Optional[str] = None, intensity: float = 0.5,
                       seed: Optional[int] = None, steps: Optional[int] = None,
                       width: Optional[int] = None, height: Optional[int] = None,
                       image_model: Optional[str] = None, wait: bool = True,
                       poll_interval: float = 2.0, on_status=None) -> Dict[str, Any]:
        """One neuromodulated image (``task="image"``). Returns the worker's result dict.

        The interesting field is ``image`` — a ``data:image/png;base64,...`` URL. The SD model is
        chosen by the endpoint's ``IMAGE_MODEL`` env; ``image_model`` overrides it only within the
        worker's allow-list. Uses the async ``/run`` + poll flow by default so a cold SD load can't
        blow the ``/runsync`` window.
        """
        payload: Dict[str, Any] = {"task": "image", "prompt": prompt,
                                   "pack_name": pack_name, "intensity": intensity}
        for k, v in (("seed", seed), ("steps", steps), ("width", width),
                     ("height", height), ("image_model", image_model)):
            if v is not None:
                payload[k] = v
        raw = self._run_async(payload, poll_interval=poll_interval, on_status=on_status) if wait \
            else self._runsync(payload)
        out = self._extract_output(raw)
        if isinstance(out, dict) and out.get("error"):
            raise RuntimeError(f"RunPod image error: {out['error']}")
        return out

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
