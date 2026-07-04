"""Lean neuromodulated Stable Diffusion interface for the serverless worker.

A trimmed sibling of ``demo/image_generation_demo.py``: it reuses the *same* neuromod machinery
(pack -> generation params + real UNet activation steering) but drops the matplotlib / spectral
analysis so it imports cleanly on an inference worker. Given a prompt + optional pack, it returns
a PNG as a base64 data URL for the chat UI to drop inline.

Deployment (chosen: co-resident on the LLM endpoints, bigger GPUs):
  * The SD pipeline loads lazily on the first ``task:"image"`` request and is cached for the life
    of the warm worker, alongside the LLM. Size the endpoint's GPU for BOTH (the small tier's
    SDXL-Turbo is ~8GB; the large tier's SDXL base is ~7GB + refiner ~7GB).
  * The model is chosen per endpoint via the ``IMAGE_MODEL`` env var (small tier default
    ``sdxl-turbo``; set the large endpoint to ``stabilityai/stable-diffusion-xl-base-1.0``).
  * Set ``IMAGE_REFINER`` (e.g. ``stabilityai/stable-diffusion-xl-refiner-1.0``) to add a refine
    pass on the large tier. ``IMAGE_CPU_OFFLOAD=1`` pages SD weights to CPU between steps if the
    GPU is tight (slower, but fits smaller cards).

Torch / diffusers are imported lazily inside the class so this module (and its pure helpers,
which the handler and tests use) import without a GPU.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Friendly alias -> HF repo id. Mirrors demo COMMON_MODELS; the small/large defaults the demo
# frontend offers. A client-supplied image model is resolved through this allow-list (like the
# text MODEL_REGISTRY) so a browser can't make a scale-to-zero GPU pull an arbitrary checkpoint.
IMAGE_MODEL_REGISTRY = {
    "sd-v1-5": "runwayml/stable-diffusion-v1-5",
    "sd-v2-1": "stabilityai/stable-diffusion-2-1",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-base": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
}

# Sensible per-tier default if IMAGE_MODEL is unset: the small, fast, 16GB-friendly one.
DEFAULT_IMAGE_MODEL = "stabilityai/sdxl-turbo"


def resolve_image_model(name: Optional[str], default: Optional[str] = None) -> str:
    """Resolve an alias/id to a vetted HF repo id (allow-list, like the text handler).

    Empty -> ``default`` (or the env/global default); a known alias -> its id; an already-known
    id -> itself. Anything else falls back to the default unless ``ALLOW_ANY_MODEL=1``.
    """
    default = default or os.environ.get("IMAGE_MODEL") or DEFAULT_IMAGE_MODEL
    default = IMAGE_MODEL_REGISTRY.get(default, default)
    if not name:
        return default
    name = str(name).strip()
    if name in IMAGE_MODEL_REGISTRY:
        return IMAGE_MODEL_REGISTRY[name]
    allowed = set(IMAGE_MODEL_REGISTRY.values())
    if name in allowed or os.environ.get("ALLOW_ANY_MODEL") == "1":
        return name
    return default


def pipeline_kind(model_name: str) -> str:
    """Classify a model id into a diffusers pipeline family: 'sdxl-turbo' | 'sdxl' | 'sd'."""
    n = (model_name or "").lower()
    if "turbo" in n:
        return "sdxl-turbo"
    if "xl" in n:
        return "sdxl"
    return "sd"


def is_turbo(model_name: str) -> bool:
    return pipeline_kind(model_name) == "sdxl-turbo"


def pil_to_data_url(image) -> str:
    """Encode a PIL image as a ``data:image/png;base64,...`` URL (inline-able in an <img>)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class NeuromodImageInterface:
    """Cache-once SD pipeline + neuromod pack adaptation for one worker."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = resolve_image_model(model_name)
        self.pipeline = None
        self.refiner = None
        self.registry = None
        self.neuromod_tool = None
        self._steer = None            # active UNet activation-steering config (None = off)
        self.generation_params: Dict[str, Any] = {}
        self.device = "cpu"
        self._load()

    # -- loading -----------------------------------------------------------------------------
    def _load(self):
        import torch
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"[image] GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("[image] CPU (no GPU) — generation will be slow")

        kind = pipeline_kind(self.model_name)
        pipe_cls = StableDiffusionXLPipeline if kind in ("sdxl", "sdxl-turbo") else StableDiffusionPipeline
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info(f"[image] Loading {self.model_name} ({kind}) dtype={dtype}")
        self.pipeline = pipe_cls.from_pretrained(
            self.model_name, torch_dtype=dtype, use_safetensors=True,
            safety_checker=None, requires_safety_checker=False,
        )
        self._place(self.pipeline)

        # Optional SDXL refine pass (large tier). Best-effort: a load failure degrades to base-only.
        refiner_id = os.environ.get("IMAGE_REFINER")
        if refiner_id and kind in ("sdxl", "sdxl-turbo"):
            try:
                from diffusers import StableDiffusionXLImg2ImgPipeline
                logger.info(f"[image] Loading refiner {refiner_id}")
                self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    refiner_id, torch_dtype=dtype, use_safetensors=True,
                )
                self._place(self.refiner)
            except Exception as e:  # pragma: no cover - depends on GPU/weights
                logger.warning(f"[image] Refiner unavailable, using base only: {e}")
                self.refiner = None

        self._setup_neuromod()

    def _place(self, pipe):
        """Move a pipeline to the device with the memory optimizations small GPUs need."""
        import torch
        if os.environ.get("IMAGE_CPU_OFFLOAD") == "1" and self.device == "cuda":
            # Page weights CPU<->GPU per step: fits tight VRAM (co-resident with a big LLM) at a
            # latency cost. Don't also call .to(cuda) — offload manages placement itself.
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        if self.device == "cuda":
            pipe.enable_attention_slicing()
            for opt in ("enable_vae_slicing", "enable_vae_tiling"):
                try:
                    getattr(pipe, opt)()
                except Exception:
                    pass
            # SDXL's VAE is numerically unstable in fp16 (black/NaN images); let diffusers upcast.
            try:
                pipe.vae.config.force_upcast = True
            except Exception:
                pass

    def _setup_neuromod(self):
        import torch
        from neuromod import NeuromodTool
        from neuromod.pack_system import PackRegistry

        cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "packs", "config.json")
        self.registry = PackRegistry(cfg if os.path.exists(cfg) else "packs/config.json")
        # Merge image-focused packs if present (adds visual effects on top of the text packs).
        try:
            import json
            from neuromod.pack_system import Pack
            ip = os.path.join(os.path.dirname(cfg), "image_focused_packs.json")
            if os.path.exists(ip):
                with open(ip) as fh:
                    for name, data in json.load(fh).get("packs", {}).items():
                        try:
                            self.registry.packs[name] = Pack.from_dict(data)
                        except Exception as e:
                            logger.debug(f"[image] skip pack {name}: {e}")
        except Exception as e:
            logger.debug(f"[image] image-focused packs not merged: {e}")
        # The neuromod tool only needs a stand-in model/tokenizer here — image effects act on the
        # UNet + sampler params, not on this dummy (same pattern as the demo).
        dummy_model = torch.nn.Linear(4, 4)
        dummy_tok = type("T", (), {"eos_token_id": 0, "pad_token_id": 0})()
        self.neuromod_tool = NeuromodTool(self.registry, dummy_model, dummy_tok)

    # -- pack adaptation ---------------------------------------------------------------------
    def _adapt_pack(self, pack_name: str, intensity: float) -> bool:
        """Apply a pack: build the UNet-steering config + sampler params. Returns applied?"""
        try:
            res = self.neuromod_tool.apply(pack_name, intensity=intensity)
        except Exception as e:
            logger.warning(f"[image] pack '{pack_name}' apply failed: {e}")
            return False
        if not (res and res.get("ok", True)):
            return False
        pack = self.neuromod_tool.registry.get_pack(pack_name)
        if pack is None:
            return False
        effects = getattr(pack, "effects", None) or []

        # The real mechanism: UNet activation steering (directional steer + entropy injection),
        # dose-scaled per pack. Sampler-knob tweaks below are a secondary nudge distilled models
        # (Turbo) largely ignore.
        steer_scale = noise_scale = 0.0
        for eff in effects:
            etype = getattr(eff, "effect", "") or (eff.get("effect", "") if isinstance(eff, dict) else "")
            w = (getattr(eff, "weight", None) if not isinstance(eff, dict) else eff.get("weight", 0.0))
            w = (w or 0.0) * intensity
            if etype == "steering":
                steer_scale += w
            elif etype in ("temperature", "entropy", "exponential_decay_kv"):
                noise_scale += w
        try:
            from neuromod.visual_steering import stable_seed
            self._steer = {
                "steer_scale": float(min(1.0, steer_scale)),
                "noise_scale": float(min(1.0, noise_scale)),
                "direction_seed": stable_seed(pack_name),
            }
        except Exception as e:
            logger.warning(f"[image] steering config unavailable: {e}")
            self._steer = None

        turbo = is_turbo(self.model_name)
        base = ({"num_inference_steps": 1} if turbo
                else {"guidance_scale": 7.5, "num_inference_steps": 40})
        try:
            from neuromod.visual_effects import apply_visual_effects_to_generation
            self.generation_params = apply_visual_effects_to_generation(effects, dict(base))
        except Exception:
            self.generation_params = dict(base)
        # Clamp to sane ranges.
        gp = self.generation_params
        if turbo:
            gp["num_inference_steps"] = max(1, min(10, int(gp.get("num_inference_steps", 1))))
            gp.pop("guidance_scale", None)
        else:
            gp["num_inference_steps"] = max(10, min(80, int(gp.get("num_inference_steps", 40))))
            if "guidance_scale" in gp:
                gp["guidance_scale"] = max(1.0, min(20.0, float(gp["guidance_scale"])))
        return True

    def clear(self):
        if self.neuromod_tool:
            try:
                self.neuromod_tool.clear()
            except Exception:
                pass
        self._steer = None
        self.generation_params = {}

    # -- generation --------------------------------------------------------------------------
    def generate(self, prompt: str, pack_name: Optional[str] = None, intensity: float = 0.5,
                 width: Optional[int] = None, height: Optional[int] = None,
                 steps: Optional[int] = None, guidance_scale: Optional[float] = None,
                 seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate one image; return ``{image: data-url, ...}`` (honest ``pack_applied``)."""
        import torch
        from contextlib import nullcontext

        start = time.time()
        turbo = is_turbo(self.model_name)
        neuromod_applied = None
        neuromod_error = None
        self.clear()
        if pack_name:
            neuromod_applied = self._adapt_pack(pack_name, intensity)
            if not neuromod_applied:
                neuromod_error = f"pack '{pack_name}' did not apply"

        gp = dict(self.generation_params) if self.generation_params else (
            {"num_inference_steps": 2} if turbo else {"num_inference_steps": 40, "guidance_scale": 7.5})
        # Explicit request overrides win over pack-derived params.
        if steps is not None:
            gp["num_inference_steps"] = int(steps)
        if guidance_scale is not None and not turbo:
            gp["guidance_scale"] = float(guidance_scale)
        gp.pop("eta", None); gp.pop("strength", None)  # not accepted by a txt2img __call__
        if turbo:
            # SDXL-Turbo is a distilled/guidance-free model: the pipeline default (5.0) degrades
            # it, so force CFG off. (num_inference_steps stays low, set by the pack/defaults.)
            gp["guidance_scale"] = 0.0

        # Resolution: SDXL is native 1024, SD1.x/Turbo 512.
        default_res = 1024 if pipeline_kind(self.model_name) == "sdxl" else 512
        w = int(width or default_res)
        h = int(height or default_res)

        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device if self.device == "cuda" else "cpu").manual_seed(int(seed))

        # Wrap denoising in UNet activation steering when a pack set a non-zero config.
        steer_ctx = nullcontext()
        s = self._steer
        if s and (s.get("steer_scale") or s.get("noise_scale")) and hasattr(self.pipeline, "unet"):
            try:
                from neuromod.visual_steering import UNetActivationSteering
                steer_ctx = UNetActivationSteering(self.pipeline.unet, **s)
            except Exception as e:
                logger.warning(f"[image] steering context failed: {e}")

        try:
            with torch.no_grad(), steer_ctx:
                out = self.pipeline(prompt=prompt, width=w, height=h, generator=gen, **gp)
                image = out.images[0]
                if self.refiner is not None:
                    image = self.refiner(prompt=prompt, image=image, generator=gen).images[0]
        finally:
            self.clear()

        return {
            "image": pil_to_data_url(image),
            "prompt": prompt,
            "pack_applied": pack_name if neuromod_applied else None,
            "neuromod_applied": neuromod_applied,
            "neuromod_error": neuromod_error,
            "intensity": intensity,
            "image_model": self.model_name,
            "width": w, "height": h,
            "generation_time": round(time.time() - start, 3),
        }
