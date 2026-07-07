#!/usr/bin/env python3
"""
Thread D -- "Vitals monitor": pharmacodynamics slider video + HTML (issue #10).

The live-demo centerpiece. A single fixed seed, one pack, swept on a fine dose grid: the
audience watches the image disintegrate frame-by-frame next to a real-time "vitals" panel
(CLIP semantic drift dropping, spectral energy exploding, LPIPS-step bars). Exports:

  * a **GIF** (always) and an **mp4** (best-effort, if an ffmpeg writer is available) for talks;
  * a **self-contained HTML slider** (`docs/`-ready, no server) that scrubs the dose and
    updates the image + a synced metric chart.

Frame generation (real images/latents) happens on a GPU via :func:`frames_from_interface`;
the renderer/exporters are pure and run anywhere. ``--demo`` fabricates synthetic frames so
the whole pipeline (composite frames, GIF, HTML) can be produced and verified without a GPU.

Usage
-----
    # GPU: real sweep of the LSD pack on one seed
    python demo/vitals_monitor.py --pack lsd --prompt "a tree" --seed 42 \
        --steps 0.0,0.04,0.08,...,1.0 --outdir outputs/vitals

    # No GPU: synthetic demo to validate the exporters
    python demo/vitals_monitor.py --demo --outdir outputs/vitals_demo
"""

import argparse
import base64
import io
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

# Ensure the repo root is importable so `from demo.image_generation_demo import ...` works
# even when this file is run directly as `python demo/vitals_monitor.py`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

METRIC_LABELS = {
    "clip_prompt_similarity": "CLIP → prompt (semantic anchor)",
    "latent_energy": "Spectral energy (entropy)",
    "lpips_step": "LPIPS step (Δ from previous dose)",
}


# --------------------------------------------------------------------------------------
# Frame model
# --------------------------------------------------------------------------------------


class Frame:
    """One dose step: an image plus its scalar metrics."""

    def __init__(self, dose: float, image, metrics: Dict[str, float]):
        self.dose = float(dose)
        self.image = image
        self.metrics = metrics


def _to_pil(image):
    from PIL import Image
    if hasattr(image, "convert"):
        return image
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8) if arr.max() > 1.0 else (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def series(frames: Sequence[Frame], metric: str) -> List[float]:
    return [f.metrics.get(metric, np.nan) for f in frames]


# --------------------------------------------------------------------------------------
# Composite frame rendering + GIF/mp4
# --------------------------------------------------------------------------------------


def render_composite(frames: Sequence[Frame], index: int, metrics: Sequence[str]):
    """Render one composite frame: image panel + synced vitals plot up to ``index``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = frames[index]
    fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(9, 4.2),
                                          gridspec_kw={"width_ratios": [1, 1.2]})
    ax_img.imshow(_to_pil(f.image))
    ax_img.axis("off")
    ax_img.set_title(f"dose = {f.dose:.2f}")

    doses = [fr.dose for fr in frames]
    for m in metrics:
        ys = np.array(series(frames, m), dtype=float)
        finite = ys[np.isfinite(ys)]
        if finite.size == 0:
            continue
        lo, hi = float(np.min(finite)), float(np.max(finite))
        norm = (ys - lo) / (hi - lo) if hi > lo else np.zeros_like(ys)
        ax_plot.plot(doses[:index + 1], norm[:index + 1], marker="o", ms=3,
                     label=METRIC_LABELS.get(m, m))
    ax_plot.axvline(f.dose, color="k", lw=1, ls="--", alpha=0.5)
    ax_plot.set_xlim(min(doses), max(doses))
    ax_plot.set_ylim(-0.05, 1.05)
    ax_plot.set_xlabel("Dose (intensity)")
    ax_plot.set_ylabel("normalized vital")
    ax_plot.legend(fontsize=7, loc="upper left")
    ax_plot.set_title("Vitals")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    return Image.open(buf).convert("RGB")


def export_animation(frames: Sequence[Frame], out_prefix: str,
                     metrics: Sequence[str], fps: int = 6) -> Dict[str, Optional[str]]:
    """Write composite frames to a GIF (always) and an mp4 (best-effort)."""
    import imageio.v2 as imageio

    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    composites = [np.asarray(render_composite(frames, i, metrics)) for i in range(len(frames))]

    gif_path = f"{out_prefix}.gif"
    imageio.mimsave(gif_path, composites, duration=1.0 / fps, loop=0)

    mp4_path = f"{out_prefix}.mp4"
    written_mp4 = None
    try:
        imageio.mimsave(mp4_path, composites, fps=fps)
        written_mp4 = mp4_path
    except Exception as exc:
        print(f"[vitals] mp4 writer unavailable ({exc}); GIF written instead.")
        # Remove any partial/broken stub the failed writer may have left behind.
        if os.path.exists(mp4_path):
            try:
                os.remove(mp4_path)
            except OSError:
                pass
    return {"gif": gif_path, "mp4": written_mp4}


# --------------------------------------------------------------------------------------
# Self-contained HTML slider
# --------------------------------------------------------------------------------------


def _png_data_uri(image) -> str:
    buf = io.BytesIO()
    _to_pil(image).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_html_slider(frames: Sequence[Frame], out_html: str,
                      metrics: Sequence[str], title: str = "Visual Pharmacodynamics") -> str:
    """Emit a dependency-free HTML page: a dose slider that updates the image + vitals readout."""
    import json

    doses = [f.dose for f in frames]
    imgs = [_png_data_uri(f.image) for f in frames]
    metric_series = {m: [None if not np.isfinite(v) else float(v) for v in series(frames, m)]
                     for m in metrics}
    labels = {m: METRIC_LABELS.get(m, m) for m in metrics}

    payload = json.dumps({"doses": doses, "images": imgs, "series": metric_series, "labels": labels})
    html = _HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", payload)
    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    with open(out_html, "w") as fh:
        fh.write(html)
    return out_html


_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:16px}
 .wrap{max-width:900px;margin:0 auto}
 .row{display:flex;gap:16px;flex-wrap:wrap}
 img{max-width:380px;width:100%;border:1px solid #333;border-radius:6px;background:#000}
 .vitals{flex:1;min-width:280px}
 .bar{height:14px;background:#222;border-radius:7px;overflow:hidden;margin:4px 0 10px}
 .fill{height:100%;background:linear-gradient(90deg,#c0392b,#e67e22)}
 label{font-size:13px;color:#bbb}
 input[type=range]{width:100%}
 h1{font-size:18px} .dose{font-variant-numeric:tabular-nums;color:#e67e22}
</style></head><body><div class="wrap">
<h1>__TITLE__ &mdash; dose <span class="dose" id="dose">0.00</span></h1>
<div class="row">
 <img id="frame" alt="generation"/>
 <div class="vitals" id="vitals"></div>
</div>
<input type="range" id="slider" min="0" value="0"/>
<script>
const DATA = __DATA__;
const slider = document.getElementById('slider');
const frame = document.getElementById('frame');
const doseEl = document.getElementById('dose');
const vitals = document.getElementById('vitals');
slider.max = DATA.doses.length - 1;
function norm(arr){const v=arr.filter(x=>x!==null);if(!v.length)return arr.map(_=>0);
  const lo=Math.min(...v),hi=Math.max(...v);return arr.map(x=>x===null?0:(hi>lo?(x-lo)/(hi-lo):0));}
const normed={};for(const m in DATA.series){normed[m]=norm(DATA.series[m]);}
function render(i){
  frame.src=DATA.images[i];
  doseEl.textContent=DATA.doses[i].toFixed(2);
  let html='';
  for(const m in DATA.series){
    const raw=DATA.series[m][i];
    const pct=Math.round(normed[m][i]*100);
    html+=`<label>${DATA.labels[m]}: <b>${raw===null?'n/a':raw.toFixed(3)}</b></label>`+
          `<div class="bar"><div class="fill" style="width:${pct}%"></div></div>`;
  }
  vitals.innerHTML=html;
}
slider.addEventListener('input',()=>render(+slider.value));
render(0);
</script></div></body></html>"""


# --------------------------------------------------------------------------------------
# Frame sources
# --------------------------------------------------------------------------------------


def synthetic_frames(n: int = 21, size: int = 96, seed: int = 0) -> List[Frame]:
    """Fabricate a dose sweep for GPU-free validation of the exporters.

    The image accrues high-frequency noise with dose; CLIP-to-prompt falls, spectral energy
    rises, LPIPS step is roughly constant -- a plausible LSD-like disintegration.
    """
    rng = np.random.RandomState(seed)
    base = rng.rand(size, size, 3)
    frames = []
    prev = None
    for i in range(n):
        dose = i / (n - 1)
        noise = np.random.RandomState(1000 + i).randn(size, size, 3) * dose * 0.6
        img = np.clip((base + noise) * 255, 0, 255).astype(np.uint8)
        step = 0.0 if prev is None else float(np.mean(np.abs(img.astype(float) - prev)) / 255.0)
        frames.append(Frame(dose, img, {
            "clip_prompt_similarity": 0.32 - 0.22 * dose,     # semantic anchor decays
            "latent_energy": 100.0 + 25.0 * dose,             # entropy climbs
            "lpips_step": step,
        }))
        prev = img.astype(float)
    return frames


def frames_from_remote(endpoint_id: str, api_key: str, pack: str, prompt: str, seed: int,
                       doses: Sequence[float], steps: int = 4, size: int = 512,
                       poll_interval: float = 2.0, image_model: str = "sdxl-turbo",
                       metrics_layer=None) -> List[Frame]:  # pragma: no cover - network
    """Generate real vitals frames on a deployed RunPod worker (fixed seed, fine dose grid).

    Requests pre-VAE latents so the live graph can plot latent spectral energy alongside CLIP
    drift. Metrics are computed locally (needs the metric deps, no SD weights).
    """
    import base64
    from io import BytesIO
    from PIL import Image
    from api.runpod_client import RunPodModelInterface
    from api.image_model import latents_from_b64
    if metrics_layer is None:
        from neuromod.metrics import pharmacodynamics as metrics_layer

    client = RunPodModelInterface(endpoint_id=endpoint_id, api_key=api_key)
    frames: List[Frame] = []
    baseline_img = None
    prev_img = None
    for dose in doses:
        pack_arg = None if float(dose) == 0.0 else pack
        out = client.generate_image(prompt, pack_name=pack_arg, intensity=float(dose), seed=int(seed),
                                    steps=steps, width=size, height=size, image_model=image_model,
                                    return_latents=True, poll_interval=poll_interval)
        url = out.get("image")
        if not url:
            continue
        b64 = url.split(",", 1)[1] if "," in url else url
        img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        latents = latents_from_b64(out["latents"]) if out.get("latents") else None
        if float(dose) == 0.0:
            baseline_img = img
        m = metrics_layer.compute_image_metrics(img, latents=latents, prompt=prompt,
                                                baseline_image=baseline_img, prev_image=prev_img)
        frames.append(Frame(dose, img, m))
        prev_img = img
    return frames


def frames_from_interface(interface, pack: str, prompt: str, seed: int,
                          doses: Sequence[float], metrics_layer=None) -> List[Frame]:  # pragma: no cover - GPU
    """Generate real frames via an ``ImageNeuromodInterface`` on a fixed seed.

    Re-seeds before every dose so only the neuromodulation changes, computes the metric
    bundle per generation, and returns Frames. Requires torch/diffusers + a loaded model.
    """
    import torch
    if metrics_layer is None:
        from neuromod.metrics import pharmacodynamics as metrics_layer

    frames: List[Frame] = []
    baseline_img = None
    prev_img = None
    for dose in doses:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        pack_arg = None if float(dose) == 0.0 else pack
        result = interface.generate_image(prompt, pack_name=pack_arg, intensity=float(dose))
        if not result.get("success"):
            continue
        img = result["image"]
        if float(dose) == 0.0:
            baseline_img = img
        m = metrics_layer.compute_image_metrics(
            img, latents=result.get("latents"), prompt=prompt,
            baseline_image=baseline_img, prev_image=prev_img)
        frames.append(Frame(dose, img, m))
        prev_img = img
    return frames


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _default_metrics(frames: Sequence[Frame]) -> List[str]:
    present = set()
    for f in frames:
        present.update(f.metrics.keys())
    return [m for m in ("clip_prompt_similarity", "latent_energy", "lpips_step") if m in present]


def build_all(frames: List[Frame], outdir: str, title: str) -> Dict[str, Optional[str]]:
    metrics = _default_metrics(frames)
    anim = export_animation(frames, os.path.join(outdir, "vitals_monitor"), metrics)
    html = build_html_slider(frames, os.path.join(outdir, "vitals_slider.html"), metrics, title=title)
    return {**anim, "html": html}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Visual pharmacodynamics vitals monitor")
    ap.add_argument("--demo", action="store_true", help="Synthetic frames (no GPU)")
    ap.add_argument("--outdir", default="outputs/vitals")
    ap.add_argument("--pack", default="lsd")
    ap.add_argument("--prompt", default="a tree")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="sdxl-turbo")
    ap.add_argument("--steps", default=None, help="Comma-separated doses (default 0..1 by 0.05)")
    ap.add_argument("--remote", action="store_true",
                    help="Generate frames on a RunPod worker (task=image) instead of a local model")
    ap.add_argument("--local", action="store_true",
                    help="Force a LOCAL Stable Diffusion load (downloads ~10GB, runs on CPU/GPU here). "
                         "Off by default so a bare invocation never silently pulls the full model.")
    ap.add_argument("--endpoint", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    ap.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    ap.add_argument("--diffusion-steps", type=int, default=4, help="Diffusion steps/image (remote)")
    ap.add_argument("--image-size", type=int, default=512, help="Square image size (remote)")
    args = ap.parse_args(argv)

    if args.steps:
        doses = [float(x) for x in args.steps.split(",") if x.strip() != ""]
    else:
        doses = [round(0.05 * i, 2) for i in range(21)]

    # Route frame generation. Prefer the RunPod worker whenever creds are available, so a bare
    # invocation NEVER silently downloads the full ~10GB SD model. Local generation must be opted
    # into explicitly with --local.
    use_remote = args.remote or (not args.local and not args.demo and args.endpoint and args.api_key)
    if args.demo:
        frames = synthetic_frames(n=len(doses))
        title = "Visual Pharmacodynamics (demo)"
    elif use_remote:  # pragma: no cover - network
        if not args.endpoint or not args.api_key:
            ap.error("--remote needs --endpoint/--api-key (or RUNPOD_ENDPOINT_ID/RUNPOD_API_KEY)")
        frames = frames_from_remote(args.endpoint, args.api_key, args.pack, args.prompt, args.seed,
                                    doses, steps=args.diffusion_steps, size=args.image_size,
                                    image_model=args.model)
        title = f"Visual Pharmacodynamics — {args.pack}"
    elif args.local:  # pragma: no cover - GPU path
        from demo.image_generation_demo import ImageNeuromodInterface
        iface = ImageNeuromodInterface(model_name=args.model)
        frames = frames_from_interface(iface, args.pack, args.prompt, args.seed, doses)
        title = f"Visual Pharmacodynamics — {args.pack}"
    else:
        ap.error(
            "No frame source selected. This tool will NOT silently download the full SD model.\n"
            "  * --remote  : generate on your RunPod worker (set RUNPOD_ENDPOINT_ID/RUNPOD_API_KEY) [recommended]\n"
            "  * --demo    : synthetic frames, no model, for testing the exporters\n"
            "  * --local   : force a local SD load here (~10GB download, slow on CPU)")

    out = build_all(frames, args.outdir, title)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
