"""
Real latent-space neuromodulation for diffusion models.

The text-pack -> sampler-knob mapping (guidance_scale/eta/steps) is NOT genuine
neuromodulation and barely moves a distilled model like SDXL-Turbo. This module implements
the actual mechanism the paper claims: it perturbs the diffusion model's *internal
activations* during denoising via UNet forward hooks, dose-dependently, with two components:

  * **Directional steering** -- add a per-channel steering vector (a fixed unit direction,
    deterministic per pack) to UNet block activations, scaled by intensity. This is the
    diffusion analog of LLM activation steering: it pushes the representation along a
    consistent direction, loosening the prompt's attractor.
  * **Entropy injection** -- add Gaussian noise to the same activations, scaled by intensity.
    This is the "temperature"/associative-flux component.

Both are scaled relative to each activation's own magnitude so the effect is stable and
interpretable (a fraction of the activation norm). Hooks are removed on context exit.

The scaling math is a pure function (:func:`apply_steer`) unit-tested without torch; the hook
manager lazily imports torch and attaches to a live UNet.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np


def unit_direction(num_channels: int, seed: int) -> np.ndarray:
    """Deterministic unit steering vector of length ``num_channels`` for a given seed."""
    rng = np.random.RandomState(seed % (2**31 - 1))
    v = rng.randn(num_channels)
    n = np.linalg.norm(v)
    return v / (n + 1e-8)


def apply_steer(x, direction, steer_scale, noise, noise_scale):
    """Core steering update: ``x + steer_scale*direction + noise_scale*noise``.

    Works on numpy arrays or torch tensors (duck-typed). ``direction`` and ``noise`` must be
    broadcastable to ``x``. Pure and unit-tested.
    """
    out = x
    if steer_scale:
        out = out + steer_scale * direction
    if noise_scale and noise is not None:
        out = out + noise_scale * noise
    return out


class UNetActivationSteering:
    """Context manager that steers a diffusers UNet's activations during generation.

    Parameters
    ----------
    unet : the pipeline's UNet (``pipeline.unet``).
    steer_scale : directional steering strength (fraction of activation magnitude at
        intensity 1). 0 disables directional steering.
    noise_scale : entropy-injection strength (fraction of activation magnitude). 0 disables.
    direction_seed : seed making the steering direction deterministic per pack.
    targets : which block groups to hook -- any of ``"mid"``, ``"up"``, ``"down"``.
    """

    def __init__(self, unet, steer_scale: float = 0.0, noise_scale: float = 0.0,
                 direction_seed: int = 0, targets: Sequence[str] = ("mid", "up")):
        self.unet = unet
        self.steer_scale = float(steer_scale)
        self.noise_scale = float(noise_scale)
        self.direction_seed = int(direction_seed)
        self.targets = tuple(targets)
        self._handles: List = []

    @property
    def active(self) -> bool:
        return self.steer_scale != 0.0 or self.noise_scale != 0.0

    def _select_modules(self) -> List:
        unet = self.unet
        mods: List = []
        if "down" in self.targets and hasattr(unet, "down_blocks"):
            mods += list(unet.down_blocks)
        if "mid" in self.targets and getattr(unet, "mid_block", None) is not None:
            mods.append(unet.mid_block)
        if "up" in self.targets and hasattr(unet, "up_blocks"):
            mods += list(unet.up_blocks)
        return mods

    def _make_hook(self, idx: int):
        import torch

        def hook(module, inputs, output):
            is_tuple = isinstance(output, tuple)
            x = output[0] if is_tuple else output
            if not hasattr(x, "dim") or x.dim() < 2:
                return output
            channels = x.shape[1]
            direction = unit_direction(channels, self.direction_seed + idx)
            shape = (1, channels) + (1,) * (x.dim() - 2)
            dvec = torch.as_tensor(direction, dtype=x.dtype, device=x.device).view(*shape)
            # Scale relative to this activation's magnitude for stability across blocks.
            mag = x.detach().abs().mean()
            noise = torch.randn_like(x) if self.noise_scale else None
            steered = apply_steer(
                x, dvec, self.steer_scale * mag, noise, self.noise_scale * mag)
            if is_tuple:
                return (steered,) + tuple(output[1:])
            return steered

        return hook

    def __enter__(self):
        if not self.active:
            return self
        import torch  # noqa: F401 - ensure torch present before hooking

        for idx, module in enumerate(self._select_modules()):
            self._handles.append(module.register_forward_hook(self._make_hook(idx)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []
        return False


def stable_seed(text: str) -> int:
    """Deterministic small int seed from a pack name (hash() is salted per-process)."""
    h = 0
    for ch in text or "":
        h = (h * 131 + ord(ch)) % (2**31 - 1)
    return h


__all__ = ["unit_direction", "apply_steer", "UNetActivationSteering", "stable_seed"]
