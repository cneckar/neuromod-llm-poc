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
    verbose: bool = True,
):
    """Execute the dose-response sweep and stream results to ``csv_path``.

    For every (pack, intensity, seed) we compute the per-generation metric bundle
    (relative to that seed's dose-0 baseline and its previous dose), then per
    (pack, intensity) we compute the inter-seed diversity across all seeds.
    """
    done = _load_done_keys(csv_path)
    clip = pdm.CLIPScorer() if pdm.clip_available() else None
    lp = pdm.LPIPSScorer() if pdm.lpips_available() else None
    fh, writer = _open_writer(csv_path)

    total = len(packs) * len(intensities) * len(seeds)
    count = 0
    try:
        for pack in packs:
            # For diversity, retain images per dose across seeds.
            images_by_dose: Dict[float, List] = {i: [] for i in intensities}
            for seed in seeds:
                baseline_image = None
                prev_image = None
                for intensity in sorted(intensities):
                    count += 1
                    key = (pack, float(intensity), int(seed))
                    result = generator.generate(pack, intensity, seed)
                    if not result.get("success", False) or result.get("image") is None:
                        if verbose:
                            print(f"  [skip] {pack} i={intensity} seed={seed}: generation failed")
                        continue
                    image = result["image"]
                    latents = result.get("latents")

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

                    _write_metrics(writer, pack, float(intensity), int(seed), metrics)
                    fh.flush()
                    prev_image = image
                    if verbose and count % 10 == 0:
                        print(f"  [{count}/{total}] {pack} i={intensity} seed={seed} "
                              f"({len(metrics)} metrics)")

            # Inter-seed diversity per dose (mode-collapse metric).
            for intensity, imgs in images_by_dose.items():
                if len(imgs) < 2:
                    continue
                div = pdm.pairwise_diversity(imgs, method=diversity_method, lpips_model=lp)
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
    ap.add_argument("--seeds", type=int, default=100, help="Number of fixed seeds (0..N-1)")
    ap.add_argument("--seed-start", type=int, default=0, help="First seed value")
    ap.add_argument("--out", default="outputs/dose_response/results.csv", help="Output CSV path")
    ap.add_argument("--concepts", default=None,
                    help="Comma-separated off-prompt concepts for CLIP proximity (Latent Specter)")
    ap.add_argument("--diversity-method", default="auto", choices=["auto", "lpips", "ssim", "l2"])
    ap.add_argument("--dry-run", action="store_true",
                    help="Use synthetic generator (no torch/model) to validate plumbing")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    packs = [x.strip() for x in args.packs.split(",") if x.strip()]
    intensities = _parse_floats(args.intensities) if args.intensities else DEFAULT_INTENSITIES
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    concepts = [x.strip() for x in args.concepts.split(",")] if args.concepts else None

    if args.dry_run:
        generator = DryRunGenerator(prompt=args.prompt)
        print("[dry-run] Using synthetic generator (no model loaded).")
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
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
