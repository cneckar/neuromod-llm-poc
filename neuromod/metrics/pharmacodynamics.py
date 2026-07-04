"""
Quantitative pharmacodynamic metrics for neuromodulated image generation.

This module turns the existing ``neuromod.apply(pack, intensity)`` dosing knob into a
measurable *dose-response* study. It provides programmatic, scalar metrics that replace
"human eyeballs" for judging what a neuromodulation pack does to a diffusion model:

  * **CLIP semantic drift** -- how far the image has drifted from its text prompt
    (does "a tree" still look like a tree?).
  * **LPIPS** -- perceptual distance vs. a baseline / previous dose (the "break point").
  * **SSIM / MS-SSIM** -- structural similarity vs. a baseline (structural integrity).
  * **FFT scalars** -- spectral energy, spatial variance, radial high/low band ratio,
    and spectral entropy of the pixel image *and* the raw diffusion latents. This is the
    scalar reducer behind the paper's static "Table 2" (spectral statistics), now
    computable per-generation so it can be plotted against dose.
  * **Inter-seed diversity** -- mean pairwise distance across seeds at a fixed dose,
    the operational definition of "mode collapse" (the "Cocaine Crunch").

Design notes
------------
* The *frequency-domain* and *structural* metrics depend only on numpy + scikit-image and
  are always available (and unit-testable without a GPU).
* The *learned* metrics (CLIP, LPIPS) require ``torch`` and their model packages. These are
  imported lazily and degrade gracefully: if unavailable, the corresponding metrics are
  simply omitted rather than crashing the run. Use :func:`clip_available` /
  :func:`lpips_available` to check.
* The FFT reducer intentionally mirrors ``FrequencyAnalyzer.compute_fft_magnitude`` in
  ``demo/image_generation_demo.py`` (``20*log(|F|+eps)``) so the scalar "energy" is
  numerically consistent with the spectral plots already in the paper.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# ----------------------------------------------------------------------------------------
# Array coercion helpers
# ----------------------------------------------------------------------------------------

ArrayLike = Union[np.ndarray, "Any"]  # np.ndarray, PIL.Image, or torch.Tensor


def to_numpy(data: ArrayLike) -> np.ndarray:
    """Coerce a PIL image / torch tensor / ndarray to a numpy array (float or uint)."""
    # torch tensor
    if hasattr(data, "detach") and hasattr(data, "cpu"):
        return data.detach().cpu().float().numpy()
    # PIL image
    if hasattr(data, "convert") and hasattr(data, "size"):
        return np.asarray(data)
    return np.asarray(data)


def to_gray(image: ArrayLike) -> np.ndarray:
    """Return a 2D float grayscale array in [0, 255] from a PIL image / ndarray.

    Accepts HxW, HxWx3, or HxWx4 inputs. Uses luminance weights for colour images.
    """
    # PIL fast-path (matches FrequencyAnalyzer which does image.convert('L'))
    if hasattr(image, "convert") and hasattr(image, "size"):
        return np.asarray(image.convert("L")).astype(np.float64)

    arr = to_numpy(image).astype(np.float64)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # Drop alpha if present, then luminance-weight.
        arr = arr[..., :3]
        weights = np.array([0.299, 0.587, 0.114])
        return arr @ weights
    raise ValueError(f"Cannot convert array of shape {arr.shape} to grayscale")


def latents_to_channels(latents: ArrayLike) -> np.ndarray:
    """Normalize raw diffusion latents to a (C, H, W) float array.

    Handles the [1, C, H, W] tensors returned by ``generate_image`` as well as bare
    (C, H, W) arrays.
    """
    arr = to_numpy(latents).astype(np.float64)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected latents of shape [1,C,H,W] or [C,H,W], got {arr.shape}")
    return arr


# ----------------------------------------------------------------------------------------
# Frequency-domain (FFT) scalar metrics -- always available (numpy only)
# ----------------------------------------------------------------------------------------

_EPS = 1e-8


def fft_log_magnitude(plane: np.ndarray) -> np.ndarray:
    """Log-magnitude spectrum of a 2D plane.

    Mirrors ``FrequencyAnalyzer.compute_fft_magnitude`` (20*log(|F|+eps)) so scalar
    "energy" derived from this is consistent with the spectral figures in the paper.
    """
    f = np.fft.fft2(plane)
    fshift = np.fft.fftshift(f)
    return 20.0 * np.log(np.abs(fshift) + _EPS)


def _radial_profile(power: np.ndarray) -> np.ndarray:
    """Azimuthally-averaged radial profile of a centered 2D power spectrum."""
    h, w = power.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.int64)
    tbin = np.bincount(r.ravel(), power.ravel())
    nr = np.bincount(r.ravel())
    nr = np.where(nr == 0, 1, nr)
    return tbin / nr


def spectral_scalars(plane: np.ndarray, low_frac: float = 0.25) -> Dict[str, float]:
    """Reduce a single 2D plane to interpretable frequency-domain scalars.

    Returns
    -------
    dict with keys:
      ``energy``            : mean log-magnitude (the paper's "Energy (FFT)").
      ``variance``          : spatial variance of the plane (the paper's "Variance (Spatial)").
      ``high_low_ratio``    : high-frequency / low-frequency radial power ratio ("rigidity").
      ``spectral_entropy``  : Shannon entropy of the normalized power spectrum (0..1).
    """
    plane = np.asarray(plane, dtype=np.float64)
    mag = fft_log_magnitude(plane)

    energy = float(np.mean(mag))
    variance = float(np.var(plane))

    # Radial band ratio on linear power (not log) so it reflects real energy distribution.
    power = np.abs(np.fft.fftshift(np.fft.fft2(plane))) ** 2
    radial = _radial_profile(power)
    n = radial.shape[0]
    split = max(1, int(round(n * low_frac)))
    low = float(np.sum(radial[:split]))
    high = float(np.sum(radial[split:]))
    high_low_ratio = high / (low + _EPS)

    # Spectral entropy: normalized power distribution -> Shannon entropy, scaled to [0,1].
    p = power.ravel()
    total = float(np.sum(p))
    if total <= 0:
        spectral_entropy = 0.0
    else:
        p = p / total
        nz = p[p > 0]
        ent = -float(np.sum(nz * np.log(nz)))
        spectral_entropy = ent / np.log(p.size)

    return {
        "energy": energy,
        "variance": variance,
        "high_low_ratio": high_low_ratio,
        "spectral_entropy": spectral_entropy,
    }


def latent_spectral_metrics(latents: ArrayLike, low_frac: float = 0.25) -> Dict[str, float]:
    """Aggregate spectral scalars across all latent channels.

    Returns channel-averaged ``latent_energy``, ``latent_variance``,
    ``latent_high_low_ratio``, ``latent_spectral_entropy`` plus per-channel energies.
    """
    channels = latents_to_channels(latents)
    per_channel = [spectral_scalars(channels[i], low_frac=low_frac) for i in range(channels.shape[0])]

    out: Dict[str, float] = {
        "latent_energy": float(np.mean([m["energy"] for m in per_channel])),
        "latent_variance": float(np.mean([m["variance"] for m in per_channel])),
        "latent_high_low_ratio": float(np.mean([m["high_low_ratio"] for m in per_channel])),
        "latent_spectral_entropy": float(np.mean([m["spectral_entropy"] for m in per_channel])),
    }
    for i, m in enumerate(per_channel):
        out[f"latent_ch{i}_energy"] = m["energy"]
    return out


def pixel_spectral_metrics(image: ArrayLike, low_frac: float = 0.25) -> Dict[str, float]:
    """Frequency-domain scalars on the grayscale pixel image."""
    gray = to_gray(image)
    s = spectral_scalars(gray, low_frac=low_frac)
    return {
        "pixel_energy": s["energy"],
        "pixel_variance": s["variance"],
        "pixel_high_low_ratio": s["high_low_ratio"],
        "pixel_spectral_entropy": s["spectral_entropy"],
    }


# ----------------------------------------------------------------------------------------
# Structural similarity (SSIM) -- available (scikit-image)
# ----------------------------------------------------------------------------------------


def ssim(image_a: ArrayLike, image_b: ArrayLike) -> float:
    """Structural similarity between two images (grayscale). 1.0 == identical."""
    from skimage.metrics import structural_similarity as _ssim

    ga = to_gray(image_a)
    gb = to_gray(image_b)
    if ga.shape != gb.shape:
        raise ValueError(f"SSIM requires equal shapes, got {ga.shape} vs {gb.shape}")
    data_range = float(max(ga.max(), gb.max()) - min(ga.min(), gb.min())) or 1.0
    return float(_ssim(ga, gb, data_range=data_range))


def _normalized_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Perceptual-free fallback distance: RMS difference scaled to [0,1]."""
    diff = (a - b).ravel()
    rng = float(max(a.max(), b.max()) - min(a.min(), b.min())) or 1.0
    return float(np.sqrt(np.mean(diff ** 2)) / rng)


# ----------------------------------------------------------------------------------------
# Learned perceptual metrics -- lazy / optional (torch)
# ----------------------------------------------------------------------------------------


class CLIPScorer:
    """Lazy CLIP wrapper for semantic-drift and concept-proximity scoring.

    Uses ``open_clip`` (ViT-B-32 / laion2b) if available, else falls back to the
    HuggingFace ``transformers`` CLIP. Unavailable if neither torch nor a CLIP backend
    is installed; check :meth:`available`.
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: Optional[str] = None):
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._backend = None
        self._device = device
        self._model_name = model_name
        self._pretrained = pretrained

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa: F401
        except Exception:
            return False
        try:
            import open_clip  # noqa: F401
            return True
        except Exception:
            pass
        try:
            import transformers  # noqa: F401
            return True
        except Exception:
            return False

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                self._model_name, pretrained=self._pretrained
            )
            self._model = model.to(self._device).eval()
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self._model_name)
            self._backend = "open_clip"
        except Exception:
            from transformers import CLIPModel, CLIPProcessor

            hf_name = "openai/clip-vit-base-patch32"
            self._model = CLIPModel.from_pretrained(hf_name).to(self._device).eval()
            self._preprocess = CLIPProcessor.from_pretrained(hf_name)
            self._backend = "transformers"

    def _image_features(self, image: ArrayLike):
        import torch

        self._ensure()
        pil = image if hasattr(image, "convert") else _to_pil(image)
        if self._backend == "open_clip":
            tensor = self._preprocess(pil).unsqueeze(0).to(self._device)
            with torch.no_grad():
                feats = self._model.encode_image(tensor)
        else:
            inputs = self._preprocess(images=pil, return_tensors="pt").to(self._device)
            with torch.no_grad():
                feats = self._model.get_image_features(**inputs)
        return feats / feats.norm(dim=-1, keepdim=True)

    def _text_features(self, texts: Sequence[str]):
        import torch

        self._ensure()
        if self._backend == "open_clip":
            tokens = self._tokenizer(list(texts)).to(self._device)
            with torch.no_grad():
                feats = self._model.encode_text(tokens)
        else:
            inputs = self._preprocess(text=list(texts), return_tensors="pt", padding=True).to(self._device)
            with torch.no_grad():
                feats = self._model.get_text_features(**inputs)
        return feats / feats.norm(dim=-1, keepdim=True)

    def image_text_similarity(self, image: ArrayLike, text: str) -> float:
        """Cosine similarity between an image and a text prompt (semantic drift proxy)."""
        img = self._image_features(image)
        txt = self._text_features([text])
        return float((img @ txt.T).squeeze().item())

    def concept_scores(self, image: ArrayLike, concepts: Sequence[str]) -> Dict[str, float]:
        """Cosine similarity between an image and each of several concept prompts.

        Used by the Latent Specter thread to measure off-prompt concept proximity
        (e.g. proximity to "a human figure" while the prompt was "a tree").
        """
        img = self._image_features(image)
        txt = self._text_features(list(concepts))
        sims = (img @ txt.T).squeeze(0)
        return {c: float(sims[i].item()) for i, c in enumerate(concepts)}


class LPIPSScorer:
    """Lazy LPIPS wrapper (learned perceptual distance). Requires ``lpips`` + torch."""

    def __init__(self, net: str = "alex", device: Optional[str] = None):
        self._model = None
        self._device = device
        self._net = net

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa: F401
            import lpips  # noqa: F401
            return True
        except Exception:
            return False

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        import lpips

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = lpips.LPIPS(net=self._net).to(self._device).eval()

    def _to_tensor(self, image: ArrayLike):
        import torch

        arr = to_numpy(image).astype(np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = arr[..., :3]
        if arr.max() > 1.0:
            arr = arr / 255.0
        arr = arr * 2.0 - 1.0  # LPIPS expects [-1, 1]
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._device)
        return t

    def distance(self, image_a: ArrayLike, image_b: ArrayLike) -> float:
        import torch

        self._ensure()
        with torch.no_grad():
            d = self._model(self._to_tensor(image_a), self._to_tensor(image_b))
        return float(d.squeeze().item())


def _to_pil(image: ArrayLike):
    from PIL import Image

    if hasattr(image, "convert"):
        return image
    arr = to_numpy(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8) if arr.max() > 1.0 else (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def clip_available() -> bool:
    return CLIPScorer.available()


def lpips_available() -> bool:
    return LPIPSScorer.available()


def ssim_available() -> bool:
    """True if scikit-image (SSIM + the SSIM-based diversity fallback) is importable."""
    try:
        import skimage.metrics  # noqa: F401
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------------------
# Inter-seed diversity (mode-collapse) -- LPIPS if available, SSIM/L2 fallback
# ----------------------------------------------------------------------------------------


def pairwise_diversity(
    images: Sequence[ArrayLike],
    method: str = "auto",
    lpips_model: Optional[LPIPSScorer] = None,
    max_pairs: Optional[int] = None,
) -> float:
    """Mean pairwise perceptual distance across a set of images at one dose.

    High value => diverse outputs; low value => mode collapse ("Cocaine Crunch").

    method:
      ``"lpips"``  -- learned perceptual distance (needs torch+lpips).
      ``"ssim"``   -- 1 - SSIM (scikit-image only).
      ``"l2"``     -- normalized RMS pixel distance.
      ``"auto"``   -- LPIPS if available, else SSIM.
    """
    imgs = list(images)
    if len(imgs) < 2:
        return 0.0

    if method == "auto":
        # Prefer LPIPS, then SSIM, then the dependency-free L2 fallback — never crash a long run
        # just because a perceptual backend isn't installed.
        if lpips_model is not None or LPIPSScorer.available():
            method = "lpips"
        elif ssim_available():
            method = "ssim"
        else:
            method = "l2"

    if method == "lpips":
        model = lpips_model or LPIPSScorer()
        dist_fn = model.distance
    elif method == "ssim":
        dist_fn = lambda a, b: 1.0 - ssim(a, b)  # noqa: E731
    elif method == "l2":
        dist_fn = lambda a, b: _normalized_l2(to_gray(a), to_gray(b))  # noqa: E731
    else:
        raise ValueError(f"Unknown diversity method: {method}")

    dists: List[float] = []
    n = len(imgs)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if max_pairs is not None and len(pairs) > max_pairs:
        # Deterministic subsample: evenly spaced stride (no RNG, for reproducibility).
        stride = max(1, len(pairs) // max_pairs)
        pairs = pairs[::stride][:max_pairs]
    for i, j in pairs:
        dists.append(float(dist_fn(imgs[i], imgs[j])))
    return float(np.mean(dists)) if dists else 0.0


# ----------------------------------------------------------------------------------------
# Per-generation metric bundle -- what the dose-response runner calls
# ----------------------------------------------------------------------------------------


def compute_image_metrics(
    image: ArrayLike,
    latents: Optional[ArrayLike] = None,
    prompt: Optional[str] = None,
    baseline_image: Optional[ArrayLike] = None,
    prev_image: Optional[ArrayLike] = None,
    clip: Optional[CLIPScorer] = None,
    lpips_model: Optional[LPIPSScorer] = None,
    low_frac: float = 0.25,
) -> Dict[str, float]:
    """Compute all available scalar metrics for a single generation.

    Always-available metrics (pixel + latent spectral scalars, SSIM vs baseline) are
    always computed. Learned metrics (CLIP drift, LPIPS) are added only when a scorer is
    supplied or the backend is installed; otherwise they are silently omitted so a run
    never fails for lack of a GPU model.

    Parameters
    ----------
    image          : the generated PIL image / array.
    latents        : raw pre-VAE latents ([1,C,H,W] or [C,H,W]); enables latent metrics.
    prompt         : text prompt; enables CLIP semantic-drift score.
    baseline_image : dose-0 image for the same seed; enables SSIM/LPIPS-vs-baseline.
    prev_image     : previous-dose image for the same seed; enables LPIPS step distance.
    clip           : optional shared :class:`CLIPScorer`.
    lpips_model    : optional shared :class:`LPIPSScorer`.
    """
    metrics: Dict[str, float] = {}

    # Pixel frequency-domain scalars (always).
    metrics.update(pixel_spectral_metrics(image, low_frac=low_frac))

    # Latent frequency-domain scalars (if latents provided).
    if latents is not None:
        try:
            metrics.update(latent_spectral_metrics(latents, low_frac=low_frac))
        except Exception:
            pass

    # Structural integrity vs baseline (SSIM always available via skimage).
    if baseline_image is not None:
        try:
            metrics["ssim_vs_baseline"] = ssim(image, baseline_image)
        except Exception:
            pass

    # CLIP semantic drift (optional).
    scorer = clip
    if scorer is None and CLIPScorer.available():
        scorer = _shared_clip()
    if scorer is not None and prompt is not None:
        try:
            metrics["clip_prompt_similarity"] = scorer.image_text_similarity(image, prompt)
        except Exception:
            pass

    # LPIPS perceptual distances (optional).
    lp = lpips_model
    if lp is None and LPIPSScorer.available():
        lp = _shared_lpips()
    if lp is not None:
        if baseline_image is not None:
            try:
                metrics["lpips_vs_baseline"] = lp.distance(image, baseline_image)
            except Exception:
                pass
        if prev_image is not None:
            try:
                metrics["lpips_step"] = lp.distance(image, prev_image)
            except Exception:
                pass

    return metrics


# Process-wide shared scorers so we load each heavy model at most once.
_CLIP_SINGLETON: Optional[CLIPScorer] = None
_LPIPS_SINGLETON: Optional[LPIPSScorer] = None


def _shared_clip() -> Optional[CLIPScorer]:
    global _CLIP_SINGLETON
    if _CLIP_SINGLETON is None:
        _CLIP_SINGLETON = CLIPScorer()
    return _CLIP_SINGLETON


def _shared_lpips() -> Optional[LPIPSScorer]:
    global _LPIPS_SINGLETON
    if _LPIPS_SINGLETON is None:
        _LPIPS_SINGLETON = LPIPSScorer()
    return _LPIPS_SINGLETON


__all__ = [
    "to_numpy",
    "to_gray",
    "latents_to_channels",
    "fft_log_magnitude",
    "spectral_scalars",
    "latent_spectral_metrics",
    "pixel_spectral_metrics",
    "ssim",
    "CLIPScorer",
    "LPIPSScorer",
    "clip_available",
    "lpips_available",
    "pairwise_diversity",
    "compute_image_metrics",
]
