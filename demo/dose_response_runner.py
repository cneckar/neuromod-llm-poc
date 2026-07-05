#!/usr/bin/env python3
"""
Dose-response batch runner for neuromodulated image generation.

Turns the existing ``neuromod.apply(pack, intensity)`` knob into a statistically usable
dose-response study. For each (pack x intensity x seed) it generates one image, captures
the raw pre-VAE latents, and reduces the pair to scalar pharmacodynamic metrics
(CLIP semantic drift, LPIPS vs baseline / previous dose, SSIM, FFT spectral energy /
spatial variance / band ratio / entropy). Results are written as a tidy long-format CSV
so downstream analysis can plot monotonic dose-response curves with confidence intervals.

Key properties
--------------
* **Fixed seeds** eliminate stochastic noise: dose 0.0 is the sober baseline for the *same*
  seed, so every metric is a within-seed delta. The RNG is re-seeded before every call
  (mirrors the pattern in ``sweep_generation``).
* **Resumable / checkpointed**: rows are appended incrementally and already-completed
  (pack, intensity, seed) triples are skipped on restart -- essential at N=100 x 11 doses.
* **Inter-seed diversity** (the mode-collapse metric) is computed per (pack, intensity)
  once all seeds at that dose are available.
* **``--dry-run``** synthesizes images/latents with numpy instead of loading the diffusion
  model, so the CSV schema and metric plumbing can be validated end-to-end without a GPU.

Examples
--------
    # Validate the pipeline + CSV schema with no GPU / no model weights:
    python demo/dose_response_runner.py --dry-run --packs lsd,cocaine --seeds 4 \
        --intensities 0.0,0.5,1.0 --out outputs/dose_response/pilot.csv

    # Full study on a GPU box (SDXL-Turbo), N=100 seeds, 11 doses:
    python demo/dose_response_runner.py --model sdxl-turbo --packs lsd,dmt,cocaine,amphetamine \
        --prompt "a tree" --seeds 100 --out outputs/dose_response/full.csv --save-images
"""

import argparse
import csv
import importlib.util
import os
import sys
from typing import Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------------------
# Metrics import: prefer the package import (works where torch is installed); fall back to
# loading the metrics module directly by path so --dry-run works in a torch-free env
# (the parent ``neuromod`` package __init__ imports torch).
# --------------------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_metrics():
    try:
        from neuromod.metrics import pharmacodynamics as pd  # type: ignore

        return pd
    except Exception:
        path = os.path.join(_REPO_ROOT, "neuromod", "metrics", "pharmacodynamics.py")
        spec = importlib.util.spec_from_file_location("pharmacodynamics", path)
        pd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pd)
        return pd


pdm = _load_metrics()


DEFAULT_INTENSITIES = [round(0.1 * i, 2) for i in range(0, 11)]  # 0.0 .. 1.0 step 0.1
CSV_FIELDS = ["pack", "intensity", "seed", "metric", "value"]

# Metric name the SD model's OWN safety checker verdict is written under (the independent
# oracle writes its own names via SafetyOracle.evaluate()). Kept in sync with
# analysis/safety_boundary.FLAG_MODEL without importing that module here.
SAFETY_FLAG_MODEL = "safety_flag_model"


# --------------------------------------------------------------------------------------
# Image persistence (for hero-figure montages) -- optional, PIL-guarded
# --------------------------------------------------------------------------------------


def _to_pil(image):
    """Coerce a numpy array or PIL image to a PIL image (None-safe)."""
    from PIL import Image  # local import: only needed when saving/loading images

    if image is None:
        return None
    if hasattr(image, "save"):  # already a PIL image
        return image
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def image_filename(pack: str, intensity: float, seed) -> str:
    """Deterministic per-(pack, dose, seed) image filename."""
    return f"{pack}__i{float(intensity):.2f}__s{seed}.png"


def save_image(image, image_dir: str, pack: str, intensity: float, seed) -> Optional[str]:
    """Persist one generation as PNG under ``image_dir``; returns the path (or None)."""
    pil = _to_pil(image)
    if pil is None:
        return None
    os.makedirs(image_dir, exist_ok=True)
    path = os.path.join(image_dir, image_filename(pack, intensity, seed))
    pil.save(path)
    return path


def load_saved_images(image_dir: str, pack: str, intensity: float) -> Dict[int, "object"]:
    """Load ``{seed: PIL image}`` saved at a given (pack, dose) for montage builders.

    Feeds ``latent_specter.ghost_montage_from_images`` / ``mode_collapse.collapse_grid_from_images``.
    """
    from PIL import Image

    prefix = f"{pack}__i{float(intensity):.2f}__s"
    out: Dict[int, object] = {}
    if not os.path.isdir(image_dir):
        return out
    for fname in os.listdir(image_dir):
        if fname.startswith(prefix) and fname.endswith(".png"):
            seed_str = fname[len(prefix):-len(".png")]
            try:
                out[int(seed_str)] = Image.open(os.path.join(image_dir, fname)).convert("RGB")
            except (ValueError, OSError):
                continue
    return out


# --------------------------------------------------------------------------------------
# CSV helpers (resumable append)
# --------------------------------------------------------------------------------------


def _load_done_keys(csv_path: str) -> set:
    """Return the set of (pack, intensity, seed) triples already present in the CSV."""
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                done.add((row["pack"], float(row["intensity"]), int(row["seed"])))
            except (KeyError, ValueError):
                continue
    return done


def _looks_blank(image) -> bool:
    """True if an image is (near-)constant — the signature of a broken decode (black/NaN VAE)."""
    try:
        arr = np.asarray(image, dtype=np.float32)
        return float(arr.std()) < 1.0   # a real generation has plenty of pixel variance
    except Exception:
        return False


def _packs_with_diversity(csv_path: str) -> set:
    """Packs that already have their inter-seed diversity ('ALL' seed) rows written.

    A pack with diversity done AND all its cells done is fully complete and can be skipped on
    resume without regenerating any images (diversity is the only thing that needs the images)."""
    packs = set()
    if not os.path.exists(csv_path):
        return packs
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("seed") == "ALL" and row.get("metric") == "interseed_diversity":
                packs.add(row["pack"])
    return packs


def _open_writer(csv_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    fh = open(csv_path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if not exists:
        writer.writeheader()
    return fh, writer


def _write_metrics(writer, pack: str, intensity: float, seed, metrics: Dict[str, float]):
    for name, value in metrics.items():
        writer.writerow(
            {"pack": pack, "intensity": intensity, "seed": seed, "metric": name, "value": value}
        )


# --------------------------------------------------------------------------------------
# Generation backends
# --------------------------------------------------------------------------------------


class DryRunGenerator:
    """Synthetic generator: produces reproducible fake image/latents from a seed.

    The synthetic signal is intentionally *dose-dependent* -- higher intensity injects
    more high-frequency noise -- so the dry run exercises the full metric pipeline and
    the resulting CSV shows a plausible (fake) monotonic trend, validating plumbing.
    """

    def __init__(self, prompt: str, size: int = 64):
        self.prompt = prompt
        self.size = size

    def generate(self, pack: str, intensity: float, seed: int) -> Dict:
        rng = np.random.RandomState(seed)
        base = rng.rand(self.size, self.size, 3)
        # Dose-dependent high-frequency perturbation (deterministic per seed+dose).
        pert_rng = np.random.RandomState(seed * 1000 + int(intensity * 100))
        noise = pert_rng.randn(self.size, self.size, 3) * intensity * 0.5
        img = np.clip((base + noise) * 255, 0, 255).astype(np.uint8)
        latents = rng.randn(1, 4, self.size // 8, self.size // 8) * (1.0 + intensity)
        return {"image": img, "latents": latents, "success": True}


class ModelGenerator:
    """Real generator wrapping ``ImageNeuromodInterface`` (requires torch + diffusers)."""

    def __init__(self, model_name: str, prompt: str):
        import torch  # noqa: F401  (surface a clear error early if missing)
        from demo.image_generation_demo import ImageNeuromodInterface

        self.prompt = prompt
        self.iface = ImageNeuromodInterface(model_name=model_name)
        self._torch = __import__("torch")

    def _seed(self, seed: int):
        self._torch.manual_seed(seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed(seed)

    def generate(self, pack: str, intensity: float, seed: int) -> Dict:
        self._seed(seed)
        pack_arg = None if intensity == 0.0 else pack  # dose 0.0 == sober baseline
        result = self.iface.generate_image(self.prompt, pack_name=pack_arg, intensity=intensity)
        self.iface.clear_neuromodulation()
        return result


class RemoteGenerator:
    """Generator that runs image generation on a deployed RunPod worker (``task="image"``).

    Generation happens on the worker's GPU (e.g. SDXL-Turbo); metrics are computed locally by
    :func:`run`. No local SD weights are needed — only the metric deps (CLIP / LPIPS / skimage).
    Determinism comes from the ``seed`` the worker seeds its generator with, so dose 0.0 is a
    stable per-seed baseline exactly as in the local path.

    With ``latents=True`` the worker also returns the pre-VAE latents, so latent-space spectral
    metrics (``latent_*``) are computed too — full parity with a local pipeline. Every image-space +
    CLIP metric (SSIM, LPIPS, inter-seed diversity, CLIP drift, pixel energy/variance/entropy) is
    always computed.
    """

    def __init__(self, prompt: str, endpoint_id: str, api_key: str, steps: int = 4,
                 size: int = 512, poll_interval: float = 2.0, image_model: Optional[str] = None,
                 timeout: int = 3600, latents: bool = True):
        from api.runpod_client import RunPodModelInterface
        self.prompt = prompt
        self.steps = steps
        self.size = size
        self.poll_interval = poll_interval
        self.image_model = image_model
        self.want_latents = latents
        self.client = RunPodModelInterface(endpoint_id=endpoint_id, api_key=api_key, timeout=timeout)

    def generate(self, pack: str, intensity: float, seed: int) -> Dict:
        import base64
        from io import BytesIO
        pack_arg = None if intensity == 0.0 else pack  # dose 0.0 == sober baseline
        try:
            out = self.client.generate_image(
                self.prompt, pack_name=pack_arg, intensity=float(intensity), seed=int(seed),
                steps=self.steps, width=self.size, height=self.size,
                image_model=self.image_model, return_latents=self.want_latents,
                poll_interval=self.poll_interval)
        except Exception as e:  # network / worker error — mark failed so run() skips this cell
            return {"image": None, "latents": None, "success": False, "error": str(e)}
        url = out.get("image") if isinstance(out, dict) else None
        if not url:
            return {"image": None, "latents": None, "success": False,
                    "error": (out or {}).get("error", "no image returned")}
        b64 = url.split(",", 1)[1] if "," in url else url
        from PIL import Image
        img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        latents = None
        if out.get("latents"):
            try:
                from api.image_model import latents_from_b64
                latents = latents_from_b64(out["latents"])
            except Exception as e:
                print(f"  [warn] latent decode failed: {e}")
        return {"image": img, "latents": latents, "success": True,
                "pack_applied": out.get("pack_applied"), "neuromod_error": out.get("neuromod_error")}


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------


def run(
    generator,
    packs: List[str],
    intensities: List[float],
    seeds: List[int],
    prompt: str,
    csv_path: str,
    concepts: Optional[List[str]] = None,
    diversity_method: str = "auto",
    diversity_max_pairs: int = 300,
    verbose: bool = True,
    image_dir: Optional[str] = None,
    safety_oracle=None,
):
    """Execute the dose-response sweep and stream results to ``csv_path``.

    For every (pack, intensity, seed) we compute the per-generation metric bundle
    (relative to that seed's dose-0 baseline and its previous dose), then per
    (pack, intensity) we compute the inter-seed diversity across all seeds.

    Optional integrations
    ----------------------
    * ``image_dir`` -- when set, each generation is saved as a PNG (see :func:`save_image`)
      so the winning thread's hero montage can be built from disk. Images the safety oracle
      flags are **redacted** (never persisted as pixels) -- only their scores/flags survive.
    * ``safety_oracle`` -- a :class:`safety_boundary.SafetyOracle`. When provided, each
      generation is scored for NSFW/violence proximity + a 0/1 oracle flag (thread B). If the
      generator surfaces the SD model's own checker verdict under ``result['safety_flag_model']``
      that is recorded too, giving the two independent detectors the analysis needs.
    """
    done = _load_done_keys(csv_path)
    div_done = _packs_with_diversity(csv_path)  # packs whose diversity ('ALL') rows already exist
    # Warn LOUDLY about any missing metric backend up front — otherwise cells "succeed" with the
    # affected metrics silently omitted, and you only find out much later (or via a diversity crash).
    missing = []
    if not pdm.clip_available():
        missing.append("CLIP (pip install open_clip_torch)  -> clip_prompt_similarity, concept probes")
    if not pdm.lpips_available():
        missing.append("LPIPS (pip install lpips)            -> lpips_vs_baseline, lpips_step, diversity")
    if not getattr(pdm, "ssim_available", lambda: True)():
        missing.append("scikit-image (pip install scikit-image) -> ssim_vs_baseline, SSIM diversity")
    if missing:
        print("\n" + "!" * 78)
        print("WARNING: metric backends missing — these metrics will be OMITTED from the run:")
        for m in missing:
            print("  - " + m)
        print("Install them and restart for the full headline metric set (SSIM / LPIPS / diversity).")
        print("!" * 78 + "\n")
    clip = pdm.CLIPScorer() if pdm.clip_available() else None
    lp = pdm.LPIPSScorer() if pdm.lpips_available() else None
    fh, writer = _open_writer(csv_path)

    total = len(packs) * len(intensities) * len(seeds)
    count = 0
    _blank_warned = [False]   # one-shot blank-image warning (mutable for closure-free scope)
    try:
        for pack in packs:
            # A pack that already has BOTH all its cells and its diversity rows is fully complete —
            # skip it entirely on resume (no regeneration; diversity is the only thing needing images).
            pack_complete = pack in div_done and all(
                (pack, float(i), int(s)) in done for i in intensities for s in seeds)
            if pack_complete:
                count += len(intensities) * len(seeds)
                if verbose:
                    print(f"  [{pack}] fully complete — skipping (all cells + diversity present)")
                continue
            need_images = pack not in div_done  # only regenerate done cells if we still owe diversity
            # For diversity, retain images per dose across seeds.
            images_by_dose: Dict[float, List] = {i: [] for i in intensities}
            for seed in seeds:
                baseline_image = None
                prev_image = None
                for intensity in sorted(intensities):
                    count += 1
                    key = (pack, float(intensity), int(seed))
                    # Cell already recorded: skip WITHOUT regenerating unless we still need its image
                    # for this pack's diversity (i.e. diversity rows aren't written yet).
                    if key in done and not need_images:
                        continue
                    result = generator.generate(pack, intensity, seed)
                    if not result.get("success", False) or result.get("image") is None:
                        if verbose:
                            print(f"  [skip] {pack} i={intensity} seed={seed}: generation failed")
                        continue
                    image = result["image"]
                    latents = result.get("latents")

                    if not _blank_warned[0] and _looks_blank(image):
                        _blank_warned[0] = True
                        print("\n" + "!" * 78)
                        print("WARNING: generated image is (near-)BLANK — the worker's decode is likely broken")
                        print("  (e.g. SDXL fp16-VAE NaN -> black). Image-space metrics will be garbage;")
                        print("  fix/redeploy the worker before trusting this run. (latents may still be OK.)")
                        print("!" * 78 + "\n", flush=True)

                    if intensity == 0.0:
                        baseline_image = image

                    images_by_dose.setdefault(float(intensity), []).append(image)

                    if key in done:
                        prev_image = image
                        continue

                    metrics = pdm.compute_image_metrics(
                        image,
                        latents=latents,
                        prompt=prompt,
                        baseline_image=baseline_image,
                        prev_image=prev_image,
                        clip=clip,
                        lpips_model=lp,
                    )
                    # Off-prompt concept proximity (Latent Specter thread) if concepts + CLIP.
                    if concepts and clip is not None:
                        try:
                            cscores = clip.concept_scores(image, concepts)
                            for c, v in cscores.items():
                                metrics[f"concept::{c}"] = v
                        except Exception:
                            pass

                    # Safety boundary thread: independent oracle + optional model checker.
                    flagged = False
                    if safety_oracle is not None:
                        try:
                            oracle_metrics = safety_oracle.evaluate(image)
                            metrics.update(oracle_metrics)
                            flagged = bool(oracle_metrics.get("safety_flag_oracle", 0))
                        except Exception:
                            pass
                        model_flag = result.get(SAFETY_FLAG_MODEL)
                        if model_flag is not None:
                            metrics[SAFETY_FLAG_MODEL] = int(bool(model_flag))
                            flagged = flagged or bool(model_flag)

                    # Persist pixels for hero montages, but never for flagged content (redaction).
                    if image_dir and not flagged:
                        try:
                            save_image(image, image_dir, pack, float(intensity), int(seed))
                        except Exception as exc:
                            if verbose:
                                print(f"  [warn] could not save image: {exc}")

                    _write_metrics(writer, pack, float(intensity), int(seed), metrics)
                    fh.flush()
                    prev_image = image
                    if verbose and count % 10 == 0:
                        print(f"  [{count}/{total}] {pack} i={intensity} seed={seed} "
                              f"({len(metrics)} metrics)")

            # Inter-seed diversity per dose (mode-collapse metric). Capped at diversity_max_pairs
            # (deterministic subsample) — full pairwise LPIPS over N=100 images x 21 doses is
            # ~100k CPU ops and looks like a hang. Skipped if this pack's diversity is already done.
            if pack not in div_done:
                doses_sorted = sorted(images_by_dose.items())
                for di, (intensity, imgs) in enumerate(doses_sorted, 1):
                    if len(imgs) < 2:
                        continue
                    if verbose:
                        print(f"  [{pack}] diversity {di}/{len(doses_sorted)} "
                              f"(dose {intensity}, {len(imgs)} imgs, <= {diversity_max_pairs} pairs)…",
                              flush=True)
                    div = pdm.pairwise_diversity(imgs, method=diversity_method, lpips_model=lp,
                                                 max_pairs=diversity_max_pairs)
                    _write_metrics(writer, pack, float(intensity), "ALL", {"interseed_diversity": div})
                    fh.flush()
    finally:
        fh.close()

    if verbose:
        print(f"Done. Wrote metrics to {csv_path}")


def _parse_floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip() != ""]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dose-response sweep for neuromodulated image generation")
    ap.add_argument("--model", default="sdxl-turbo", help="Model name or shortcut (see COMMON_MODELS)")
    ap.add_argument("--packs", default="lsd", help="Comma-separated pack names")
    ap.add_argument("--prompt", default="a tree", help="Text prompt (held fixed across conditions)")
    ap.add_argument("--intensities", default=None,
                    help="Comma-separated doses (default 0.0..1.0 step 0.1)")
    ap.add_argument("--intensity-step", type=float, default=None,
                    help="Build a 0.0..1.0 dose grid with this step (e.g. 0.05 for the fine grid)")
    ap.add_argument("--seeds", type=int, default=100, help="Number of fixed seeds (0..N-1)")
    ap.add_argument("--seed-start", type=int, default=0, help="First seed value")
    ap.add_argument("--out", default="outputs/dose_response/results.csv", help="Output CSV path")
    ap.add_argument("--concepts", default=None,
                    help="Comma-separated off-prompt concepts for CLIP proximity (Latent Specter)")
    ap.add_argument("--diversity-method", default="auto", choices=["auto", "lpips", "ssim", "l2"])
    ap.add_argument("--diversity-max-pairs", type=int, default=300,
                    help="Cap pairwise comparisons for inter-seed diversity (deterministic subsample)")
    ap.add_argument("--image-dir", default=None,
                    help="Directory to save generated images (for hero montages; flagged ones redacted)")
    ap.add_argument("--save-images", action="store_true",
                    help="Save images under <out>_images/ (shortcut for --image-dir)")
    ap.add_argument("--safety", action="store_true",
                    help="Attach the independent SafetyOracle (thread B); needs CLIP")
    ap.add_argument("--no-sd-checker", action="store_true",
                    help="With --safety, skip the SD-native safety checker (CLIP oracle only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Use synthetic generator (no torch/model) to validate plumbing")
    # Remote generation on a deployed RunPod worker (task="image"); metrics computed locally.
    ap.add_argument("--remote", action="store_true",
                    help="Generate on a RunPod worker instead of a local pipeline (task=image)")
    ap.add_argument("--endpoint", default=os.environ.get("RUNPOD_ENDPOINT_ID"),
                    help="RunPod endpoint id (or set RUNPOD_ENDPOINT_ID)")
    ap.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"),
                    help="RunPod API key (or set RUNPOD_API_KEY)")
    ap.add_argument("--steps", type=int, default=4, help="Diffusion steps per image (remote)")
    ap.add_argument("--image-size", type=int, default=512, help="Square image size (remote)")
    ap.add_argument("--poll-interval", type=float, default=2.0, help="RunPod /status poll seconds")
    ap.add_argument("--no-latents", action="store_true",
                    help="Remote: don't fetch pre-VAE latents (drops latent_* metrics; smaller payload)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    packs = [x.strip() for x in args.packs.split(",") if x.strip()]
    if args.intensities:
        intensities = _parse_floats(args.intensities)
    elif args.intensity_step:
        n = int(round(1.0 / args.intensity_step))
        intensities = [round(i * args.intensity_step, 4) for i in range(n + 1)]
    else:
        intensities = DEFAULT_INTENSITIES
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    concepts = [x.strip() for x in args.concepts.split(",")] if args.concepts else None

    image_dir = args.image_dir
    if args.save_images and not image_dir:
        image_dir = os.path.splitext(args.out)[0] + "_images"

    safety_oracle = None
    if args.safety:
        sb_path = os.path.join(_REPO_ROOT, "analysis", "safety_boundary.py")
        spec = importlib.util.spec_from_file_location("safety_boundary", sb_path)
        sb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sb)
        # Two independent detectors, both run on the driver over the returned image: the CLIP
        # concept oracle and (unless disabled) the model's own StableDiffusionSafetyChecker.
        model_checker = None if args.no_sd_checker else sb.SDModelChecker()
        safety_oracle = sb.SafetyOracle(model_checker=model_checker)

    if args.dry_run:
        generator = DryRunGenerator(prompt=args.prompt)
        print("[dry-run] Using synthetic generator (no model loaded).")
    elif args.remote:
        if not args.endpoint or not args.api_key:
            ap.error("--remote needs --endpoint/--api-key (or RUNPOD_ENDPOINT_ID/RUNPOD_API_KEY)")
        generator = RemoteGenerator(prompt=args.prompt, endpoint_id=args.endpoint,
                                    api_key=args.api_key, steps=args.steps, size=args.image_size,
                                    poll_interval=args.poll_interval, image_model=args.model,
                                    latents=not args.no_latents)
        print(f"[remote] Generating on RunPod endpoint {args.endpoint} "
              f"(image_model={args.model}, steps={args.steps}, size={args.image_size}).")
    else:
        generator = ModelGenerator(model_name=args.model, prompt=args.prompt)

    print(f"Packs={packs} | doses={intensities} | seeds={len(seeds)} | out={args.out}")
    run(
        generator=generator,
        packs=packs,
        intensities=intensities,
        seeds=seeds,
        prompt=args.prompt,
        csv_path=args.out,
        concepts=concepts,
        diversity_method=args.diversity_method,
        diversity_max_pairs=args.diversity_max_pairs,
        verbose=not args.quiet,
        image_dir=image_dir,
        safety_oracle=safety_oracle,
    )


if __name__ == "__main__":
    main()
