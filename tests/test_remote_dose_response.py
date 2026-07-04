"""Torch-free tests for the remote (RunPod) dose-response path.

Covers the client's image-task payload + result extraction and the runner's RemoteGenerator
decode — both with a fake in place of the network, so no GPU / HTTP / SD weights are needed.
"""

import base64
import io

import pytest

from api.runpod_client import RunPodModelInterface
from demo.dose_response_runner import RemoteGenerator

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _png_data_url(color=(10, 20, 30), size=8):
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---- client.generate_image ------------------------------------------------------------------

def test_generate_image_builds_image_task_payload(monkeypatch):
    client = RunPodModelInterface(endpoint_id="ep", api_key="k")
    captured = {}

    def fake_run_async(payload, poll_interval=5.0, on_status=None):
        captured["payload"] = payload
        return {"output": [{"image": _png_data_url(), "pack_applied": "lsd", "task": "image"}]}

    monkeypatch.setattr(client, "_run_async", fake_run_async)
    out = client.generate_image("a tree", pack_name="lsd", intensity=0.5, seed=7, steps=4,
                                width=512, height=512)
    p = captured["payload"]
    assert p["task"] == "image" and p["prompt"] == "a tree" and p["pack_name"] == "lsd"
    assert p["intensity"] == 0.5 and p["seed"] == 7 and p["steps"] == 4 and p["width"] == 512
    assert out["image"].startswith("data:image/png;base64,")
    assert out["pack_applied"] == "lsd"


def test_generate_image_omits_unset_optional_params(monkeypatch):
    client = RunPodModelInterface(endpoint_id="ep", api_key="k")
    captured = {}
    monkeypatch.setattr(client, "_run_async",
                        lambda payload, **_: captured.setdefault("p", payload) or
                        {"output": [{"image": _png_data_url()}]})
    client.generate_image("x")
    p = captured["p"]
    assert "seed" not in p and "steps" not in p and "image_model" not in p  # only set when provided


def test_generate_image_raises_on_worker_error(monkeypatch):
    client = RunPodModelInterface(endpoint_id="ep", api_key="k")
    monkeypatch.setattr(client, "_run_async", lambda payload, **_: {"output": [{"error": "OOM"}]})
    with pytest.raises(RuntimeError, match="OOM"):
        client.generate_image("x")


# ---- RemoteGenerator (runner) ---------------------------------------------------------------

class FakeClient:
    def __init__(self, with_latents=False):
        self.calls = []
        self.with_latents = with_latents

    def generate_image(self, prompt, pack_name=None, intensity=0.5, seed=None, steps=None,
                       width=None, height=None, image_model=None, return_latents=False,
                       poll_interval=2.0):
        self.calls.append({"pack_name": pack_name, "intensity": intensity, "seed": seed,
                           "return_latents": return_latents})
        out = {"image": _png_data_url((intensity and 200 or 0, 0, 0)), "pack_applied": pack_name}
        if return_latents and self.with_latents:
            import numpy as np
            from api.image_model import latents_to_b64
            out["latents"] = latents_to_b64(np.zeros((1, 4, 8, 8), dtype="float32"))
        return out


def _remote_with_fake():
    gen = RemoteGenerator(prompt="a tree", endpoint_id="ep", api_key="k", steps=4, size=8)
    gen.client = FakeClient()
    return gen


def test_remote_generator_decodes_png_to_image():
    gen = _remote_with_fake()
    res = gen.generate("lsd", 0.5, 3)
    assert res["success"] is True and res["latents"] is None
    assert res["image"].size == (8, 8)          # decoded PIL image
    assert gen.client.calls[0]["pack_name"] == "lsd" and gen.client.calls[0]["seed"] == 3


def test_remote_generator_decodes_latents_when_returned():
    gen = RemoteGenerator(prompt="a tree", endpoint_id="ep", api_key="k", size=8, latents=True)
    gen.client = FakeClient(with_latents=True)
    res = gen.generate("lsd", 0.5, 3)
    assert gen.client.calls[0]["return_latents"] is True   # driver asked for latents
    assert res["latents"] is not None and res["latents"].shape == (1, 4, 8, 8)


def test_remote_generator_dose0_is_sober_baseline():
    # dose 0.0 must be generated with NO pack (pack_name=None), matching the local path.
    gen = _remote_with_fake()
    gen.generate("lsd", 0.0, 0)
    assert gen.client.calls[0]["pack_name"] is None


def test_remote_generator_marks_failure_without_raising():
    gen = _remote_with_fake()

    def boom(*a, **k):
        raise RuntimeError("network down")

    gen.client.generate_image = boom
    res = gen.generate("lsd", 0.5, 0)
    assert res["success"] is False and res["image"] is None and "network down" in res["error"]
