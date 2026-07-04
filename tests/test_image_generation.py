"""Torch-free tests for the image-generation path.

Covers the pure SD helpers (model allow-list, pipeline classification, PNG data-url) and the
RunPod handler's ``task:"image"`` routing with an injected fake image model (no GPU / diffusers).
"""

import importlib.util
import os

from api.image_model import (
    resolve_image_model, pipeline_kind, is_turbo, IMAGE_MODEL_REGISTRY, DEFAULT_IMAGE_MODEL,
)


def _load_handler():
    spec = importlib.util.spec_from_file_location(
        "rp_handler", os.path.join(os.path.dirname(__file__), "..", "deploy", "runpod", "handler.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeImageModel:
    """Stand-in for NeuromodImageInterface — echoes its args so routing is observable."""

    def __init__(self):
        self.calls = []

    def generate(self, prompt, pack_name=None, intensity=0.5, width=None, height=None,
                 steps=None, guidance_scale=None, seed=None, return_latents=False):
        self.calls.append(dict(prompt=prompt, pack_name=pack_name, intensity=intensity,
                               width=width, height=height, steps=steps, seed=seed,
                               return_latents=return_latents))
        return {
            "image": "data:image/png;base64,AAAA",
            "prompt": prompt,
            "pack_applied": pack_name,
            "neuromod_applied": bool(pack_name),
            "intensity": intensity,
            "image_model": "stabilityai/sdxl-turbo",
            "width": width or 512, "height": height or 512,
        }


# ---- pure helpers ---------------------------------------------------------------------------

def test_resolve_image_model_allowlist_and_default():
    assert resolve_image_model("sdxl-turbo") == "stabilityai/sdxl-turbo"
    assert resolve_image_model("sdxl") == "stabilityai/stable-diffusion-xl-base-1.0"
    assert resolve_image_model(None, "sdxl") == "stabilityai/stable-diffusion-xl-base-1.0"
    # Unknown id is refused (no ALLOW_ANY_MODEL) -> falls back to the default.
    os.environ.pop("ALLOW_ANY_MODEL", None)
    assert resolve_image_model("evil/huge", "sdxl-turbo") == "stabilityai/sdxl-turbo"
    # An already-vetted full id passes through.
    assert resolve_image_model("stabilityai/sdxl-turbo") == "stabilityai/sdxl-turbo"


def test_resolve_image_model_env_default(monkeypatch):
    monkeypatch.setenv("IMAGE_MODEL", "sd-v1-5")
    assert resolve_image_model(None) == "runwayml/stable-diffusion-v1-5"
    monkeypatch.delenv("IMAGE_MODEL", raising=False)
    assert resolve_image_model(None) == DEFAULT_IMAGE_MODEL


def test_pipeline_kind_and_turbo():
    assert pipeline_kind("stabilityai/sdxl-turbo") == "sdxl-turbo"
    assert pipeline_kind("stabilityai/stable-diffusion-xl-base-1.0") == "sdxl"
    assert pipeline_kind("runwayml/stable-diffusion-v1-5") == "sd"
    assert is_turbo("stabilityai/sdxl-turbo")
    assert not is_turbo("runwayml/stable-diffusion-v1-5")


# ---- handler routing ------------------------------------------------------------------------

def test_parse_event_carries_image_params():
    h = _load_handler()
    p = h.parse_event({"input": {"task": "image", "prompt": "a fox", "pack_name": "lsd",
                                 "intensity": 2.0, "width": 1024, "height": 768,
                                 "steps": 30, "seed": 7}})
    assert p["task"] == "image" and p["prompt"] == "a fox" and p["pack_name"] == "lsd"
    assert (p["width"], p["height"], p["steps"], p["seed"], p["intensity"]) == (1024, 768, 30, 7, 2.0)


def test_handler_routes_image_task_to_generate():
    h = _load_handler()
    fake = FakeImageModel()
    out = h.handler({"input": {"task": "image", "prompt": "a fox", "pack_name": "lsd",
                               "intensity": 1.5}}, model=fake)
    assert out["image"].startswith("data:image/png;base64,")
    assert out["task"] == "image" and out["pack_applied"] == "lsd"
    assert "gpu_seconds" in out
    assert fake.calls[0]["prompt"] == "a fox" and fake.calls[0]["intensity"] == 1.5


def test_image_task_requires_a_prompt():
    h = _load_handler()
    err = h.run_image(h.parse_event({"input": {"task": "image"}}), image_model=FakeImageModel())
    assert "error" in err


def test_stream_handler_yields_single_image_dict():
    h = _load_handler()
    evts = list(h.stream_handler({"input": {"task": "image", "prompt": "x"}}, model=FakeImageModel()))
    assert len(evts) == 1 and evts[0]["image"].startswith("data:image")
