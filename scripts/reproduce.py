#!/usr/bin/env python3
"""
Reproduce the Digital Psychopharmacology paper — one tiered, resumable playbook.

Regenerates the key experiments + supporting collateral behind the paper
(`outputs/DigitalPsychopharmacologyPaper/Digital-Psychopharmacology.tex`) and writes a
single **REPRODUCTION_REPORT.md** that maps every produced artifact to the figure/table/claim
it supports — so anyone cloning the repo can reproduce the findings and see exactly what
supports what.

It does not re-implement any experiment: each stage shells out to the existing script
(`scripts/…`, `analysis/…`, `demo/…`) with the right arguments, records status/outputs/runtime,
and continues on failure so a partial environment still yields a useful report.

Three compute tiers
--------------------
* **T0 (CPU-only, minutes, anyone):** validators, committed-data figures, and the whole visual
  dose-response pipeline in ``--dry-run`` / ``--demo`` / synthetic form. No GPU, no model
  weights, no tokens — proves the plumbing and regenerates everything that can be regenerated
  from data already in the repo.
* **T1 (single GPU, ungated):** real SDXL-Turbo dose-response (Table 2 / ghost / safety /
  collapse / vitals) + the text battery on **gpt2** (endpoints → Table 1 stats + Figs 2/3).
  Reproduces the paper's *methodology and qualitative* claims with open weights.
* **T2 (GPU + gated Llama-3.1-8B-Instruct, paper-scale):** the text experiments on the paper's
  own model (ablation, Lazarus, calibration, full pack set) — the only tier that reproduces the
  paper's *exact* numbers. Requires an HF token with Llama access.

Examples
--------
    # Plumbing / committed-data reproduction, no GPU:
    python scripts/reproduce.py --tier 0

    # Real headline collateral on one GPU (gpt2 text, SDXL visual):
    python scripts/reproduce.py --tier 1 --seeds 16

    # Faithful paper-scale replication on the gated model:
    HUGGINGFACE_TOKEN=hf_... python scripts/reproduce.py --tier 2 \
        --model meta-llama/Llama-3.1-8B-Instruct --seeds 100

    # Inspect the plan without running anything:
    python scripts/reproduce.py --tier 2 --list
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Pack sets. QUICK = one representative per drug class + placebo (T1 gpt2). FULL = the paper's
# 13-pack panel (T2). Override with --packs.
PACKS_QUICK = ["lsd", "cocaine", "morphine", "placebo"]
PACKS_FULL = ["lsd", "psilocybin", "mescaline", "dmt", "2c_b",
              "amphetamine", "cocaine", "methylphenidate",
              "heroin", "benzodiazepines", "morphine", "caffeine", "placebo"]


@dataclass
class Config:
    tier: int
    model: str
    seeds: int
    intensities: str
    packs: List[str]
    prompt: str
    outdir: str            # where the report/manifest + tier artifacts live
    max_tokens: int
    max_questions: int
    timeout: Optional[int]

    @property
    def figures_dir(self) -> str:
        return os.path.join(self.outdir, "figures")


@dataclass
class Stage:
    key: str
    title: str
    min_tier: int
    paper_ref: str
    modality: str                                   # validation | text | visual
    # build() -> list of argv lists to run (empty list = nothing to run for this cfg).
    build: Callable[[Config], List[List[str]]]
    outputs: Callable[[Config], List[str]]          # glob patterns produced (for the report)
    note: str = ""
    committed: bool = False                          # verify-only: surfaces data already in-repo
    requires: List[str] = field(default_factory=list)  # importable modules the stage needs


# --------------------------------------------------------------------------------------
# Stage command builders (each returns argv lists; reuse existing scripts, never reinvent)
# --------------------------------------------------------------------------------------


def _py(*args: str) -> List[str]:
    return [sys.executable, *args]


def _validation_stages() -> List[Stage]:
    return [
        Stage(
            key="blinding_audit", title="Double-blind leakage audit", min_tier=0,
            paper_ref="§4.3 Double-Blindfold Protocol", modality="validation",
            build=lambda c: [_py("scripts/audit_blinding.py", "--test-dir", "neuromod/testing",
                                 "--output-dir", os.path.join(c.outdir, "validation/blinding"))],
            outputs=lambda c: [os.path.join(c.outdir, "validation/blinding/*")],
            requires=["torch"],
        ),
        Stage(
            key="design_validation", title="Experimental-design validation", min_tier=0,
            paper_ref="§4.4 Experimental Design", modality="validation",
            build=lambda c: [_py("scripts/validate_experimental_design.py",
                                 "--output-dir", os.path.join(c.outdir, "validation/experimental_design"))],
            outputs=lambda c: [os.path.join(c.outdir, "validation/experimental_design/*")],
            requires=["torch"],
        ),
        Stage(
            key="stats_validation", title="Statistical-methods validation", min_tier=0,
            paper_ref="§4.7 Statistical Analysis", modality="validation",
            build=lambda c: [_py("scripts/validate_statistics.py", "--mock-only",
                                 "--output-dir", os.path.join(c.outdir, "validation/statistics"))],
            outputs=lambda c: [os.path.join(c.outdir, "validation/statistics/*")],
            requires=["torch"],
        ),
        Stage(
            key="power_analysis", title="Power / sample-size analysis", min_tier=0,
            paper_ref="§4.7 Power Analysis (N=126)", modality="validation",
            build=lambda c: [_py("analysis/power_analysis.py", "--plan", "analysis/plan.yaml",
                                 "--output", os.path.join(c.outdir, "power_analysis_report.json"))],
            outputs=lambda c: [os.path.join(c.outdir, "power_analysis_report.json")],
        ),
    ]


def _committed_stages() -> List[Stage]:
    return [
        Stage(
            key="ablation_committed", title="LSD ablation runs (committed data)", min_tier=0,
            paper_ref="Fig stimulant_ceiling / lsd_resistance / ablation_comparison", modality="text",
            build=lambda c: [], committed=True,
            outputs=lambda c: ["outputs/ablation_experiments/*.json"],
            note="Committed steering-vs-temperature ablation results; T2 regenerates them on Llama.",
        ),
        Stage(
            key="steering_committed", title="Steering vectors (committed)", min_tier=0,
            paper_ref="§3 Steering Vector Construction", modality="text",
            build=lambda c: [], committed=True,
            outputs=lambda c: ["outputs/steering_vectors/*.pt"],
            note="Committed MDV+PCA steering vectors (layer -1); T1 regenerates on gpt2.",
        ),
        Stage(
            key="figure4_committed", title="Cognitive-impact figure (committed)", min_tier=0,
            paper_ref="Fig cognitive (fig:cognitive)", modality="text",
            build=lambda c: [], committed=True,
            outputs=lambda c: ["outputs/figure_4_cognitive_impact.png"],
            note="No standalone generator in-repo; surfaced from the committed render.",
        ),
    ]


def _text_committed_figure_stages() -> List[Stage]:
    return [
        Stage(
            key="figure_emotion", title="Emotional-signatures radar (Fig 10)", min_tier=0,
            paper_ref="Fig emotion (fig:emotion)", modality="text",
            build=lambda c: [_py("scripts/generate_figure_5.py",
                                 "--input", "outputs/reports/emotion",
                                 "--output", os.path.join(c.figures_dir, "figure_5_emotion_signatures.png"))],
            outputs=lambda c: [os.path.join(c.figures_dir, "figure_5_emotion_signatures.png")],
            note="Runs from committed emotion results — no model needed.",
        ),
    ]


def _visual_stages() -> List[Stage]:
    def pilot_cmd(c: Config) -> List[List[str]]:
        args = ["scripts/run_pilot.py", "--seeds", str(c.seeds),
                "--intensities", c.intensities, "--prompt", c.prompt,
                "--outdir", os.path.join(c.outdir, "pilot")]
        if c.tier == 0:
            args.append("--dry-run")
        else:
            args += ["--model", "sdxl-turbo"]
        return [_py(*args)]

    def vitals_cmd(c: Config) -> List[List[str]]:
        args = ["demo/vitals_monitor.py", "--pack", "lsd", "--prompt", c.prompt,
                "--seed", "42", "--outdir", os.path.join(c.outdir, "vitals")]
        if c.tier == 0:
            args.append("--demo")
        else:
            args += ["--model", "sdxl-turbo"]
        return [_py(*args)]

    pilot_dir = lambda c: os.path.join(c.outdir, "pilot")  # noqa: E731

    return [
        Stage(
            key="visual_pilot",
            title="Cross-modal dose-response pilot + decision matrix + hero montage", min_tier=0,
            paper_ref="Fig visual_trips / Table 2 spectral / decision matrix", modality="visual",
            build=pilot_cmd,
            outputs=lambda c: [os.path.join(pilot_dir(c), "decision_matrix.csv"),
                               os.path.join(pilot_dir(c), "hero_*.png"),
                               os.path.join(pilot_dir(c), "*.csv")],
            note="T0 = --dry-run (synthetic); T1+ = real SDXL-Turbo generations.",
        ),
        Stage(
            key="dose_stats", title="Dose-response curves (CI ribbons, EC50, monotonicity)", min_tier=0,
            paper_ref="§6.2 Dose-Response & Monotonicity (Limitation → Result)", modality="visual",
            build=lambda c: [_py("analysis/dose_response_stats.py",
                                 "--in", os.path.join(pilot_dir(c), "mode_collapse.csv"),
                                 "--outdir", os.path.join(c.outdir, "dose_stats"), "--plots")],
            outputs=lambda c: [os.path.join(c.outdir, "dose_stats/*")],
        ),
        Stage(
            key="mode_collapse", title="Cocaine Crunch: stimulant mode-collapse curves", min_tier=0,
            paper_ref="Table 2 (cocaine constriction vs amphetamine agitation)", modality="visual",
            build=lambda c: [_py("analysis/mode_collapse.py",
                                 "--in", os.path.join(pilot_dir(c), "mode_collapse.csv"),
                                 "--stimulants", "cocaine,amphetamine",
                                 "--outdir", os.path.join(c.outdir, "mode_collapse"))],
            outputs=lambda c: [os.path.join(c.outdir, "mode_collapse/*")],
        ),
        Stage(
            key="latent_specter", title="Latent Specter: statistical ghost vs placebo", min_tier=1,
            paper_ref="Fig dmt_ghost (fig:dmt_ghost) → statistical generalization", modality="visual",
            build=lambda c: [_py("analysis/latent_specter.py",
                                 "--in", os.path.join(pilot_dir(c), "latent_specter.csv"),
                                 "--treatments", "dmt,lsd", "--placebo", "placebo",
                                 "--concept", "a human figure",
                                 "--outdir", os.path.join(c.outdir, "latent_specter"))],
            outputs=lambda c: [os.path.join(c.outdir, "latent_specter/*")],
            note="Needs CLIP off-prompt concept columns → T1+ only (absent in dry-run).",
        ),
        Stage(
            key="safety_boundary", title="Architectural jailbreak: safety trigger-rate vs dose", min_tier=1,
            paper_ref="§7.4 Spectral Safety Auditing (Future Work → Result)", modality="visual",
            build=lambda c: [_py("analysis/safety_boundary.py",
                                 "--in", os.path.join(pilot_dir(c), "safety_boundary.csv"),
                                 "--packs", "lsd,dmt,cocaine",
                                 "--outdir", os.path.join(c.outdir, "safety_boundary"))],
            outputs=lambda c: [os.path.join(c.outdir, "safety_boundary/*")],
            note="Needs the SafetyOracle flags written at generation time → T1+ only.",
        ),
        Stage(
            key="vitals", title="Vitals monitor: dose slider (GIF + HTML)", min_tier=0,
            paper_ref="Demo vehicle for the dose-response findings", modality="visual",
            build=vitals_cmd,
            outputs=lambda c: [os.path.join(c.outdir, "vitals/*")],
            note="T0 = --demo (synthetic); T1+ = real LSD sweep.",
        ),
    ]


def _text_experiment_stages() -> List[Stage]:
    endpoints_dir = lambda c: os.path.join(c.outdir, "endpoints")  # noqa: E731

    def endpoints_cmd(c: Config) -> List[List[str]]:
        cmds = []
        for pack in c.packs:
            args = ["scripts/calculate_endpoints.py", "--pack", pack, "--model", c.model,
                    "--output-dir", endpoints_dir(c), "--skip-completed"]
            cmds.append(_py(*args))
        return cmds

    return [
        Stage(
            key="steering_vectors", title="Generate steering vectors", min_tier=1,
            paper_ref="§3 Steering Vector Construction", modality="text",
            build=lambda c: [_py("scripts/generate_steering_vectors.py", "--model", c.model,
                                 "--dataset", "datasets/steering_prompts.jsonl",
                                 "--output-dir", os.path.join(c.outdir, "steering_vectors"),
                                 "--min-pairs", "20" if c.tier == 1 else "100")],
            outputs=lambda c: [os.path.join(c.outdir, "steering_vectors/*.pt")],
        ),
        Stage(
            key="endpoints", title="Primary-endpoint detection battery", min_tier=1,
            paper_ref="Table 1 (tab:stats) primary detection", modality="text",
            build=endpoints_cmd,
            outputs=lambda c: [os.path.join(endpoints_dir(c), "endpoints_*.json")],
            note="One run per pack. gpt2 at T1 (illustrative), Llama-3.1-8B at T2 (paper numbers).",
        ),
        Stage(
            key="analyze_endpoints", title="Endpoint statistics (paired tests, BH-FDR, effect sizes)", min_tier=1,
            paper_ref="Table 1 statistics", modality="text",
            build=lambda c: [_py("scripts/analyze_endpoints.py",
                                 "--input-dir", endpoints_dir(c),
                                 "--output-dir", os.path.join(c.outdir, "analysis"))],
            outputs=lambda c: [os.path.join(c.outdir, "analysis/*")],
        ),
        Stage(
            key="figure_detection", title="Detection-sensitivity bar chart (Fig 4)", min_tier=1,
            paper_ref="Fig sensitivity (fig:sensitivity)", modality="text",
            build=lambda c: [_py("scripts/generate_figure_2.py", "--input", endpoints_dir(c),
                                 "--output", os.path.join(c.figures_dir, "figure_2_detection_sensitivity.png"))],
            outputs=lambda c: [os.path.join(c.figures_dir, "figure_2_detection_sensitivity.png")],
        ),
        Stage(
            key="figure_radar", title="Behavioral-signature radar (Fig 6)", min_tier=1,
            paper_ref="Fig radar (fig:radar)", modality="text",
            build=lambda c: [_py("scripts/generate_figure_3.py", "--input", endpoints_dir(c),
                                 "--output", os.path.join(c.figures_dir, "figure_3_radar_plots.png"))],
            outputs=lambda c: [os.path.join(c.figures_dir, "figure_3_radar_plots.png")],
        ),
        Stage(
            key="export_ndjson", title="Export endpoints → NDJSON (power tooling)", min_tier=1,
            paper_ref="§4.7 Power Analysis input", modality="text",
            build=lambda c: [_py("scripts/export_endpoints_to_ndjson.py",
                                 "--input-dir", endpoints_dir(c),
                                 "--output", os.path.join(endpoints_dir(c), "pilot_data.jsonl"))],
            outputs=lambda c: [os.path.join(endpoints_dir(c), "pilot_data.jsonl")],
        ),
        # ---- Tier 2: paper-scale, gated-model experiments ----
        Stage(
            key="lsd_ablation", title="LSD ablation: steering vs temperature (paper-scale)", min_tier=2,
            paper_ref="Fig ablation_comparison / stimulant_ceiling / lsd_resistance", modality="text",
            build=lambda c: [_py("scripts/run_lsd_ablation_experiment.py", "--model", c.model,
                                 "--max-tokens", str(c.max_tokens))],
            outputs=lambda c: ["outputs/ablation_experiments/lsd_ablation_*.json"],
        ),
        Stage(
            key="lazarus", title="Lazarus protocol: Morphine→Cocaine resuscitation", min_tier=2,
            paper_ref="§7.3 Digital IV / bidirectional stimulant steering", modality="text",
            build=lambda c: [_py("scripts/run_lazarus_protocol_experiment.py", "--model", c.model,
                                 "--output-dir", os.path.join(c.outdir, "lazarus"))],
            outputs=lambda c: [os.path.join(c.outdir, "lazarus/*")],
        ),
        Stage(
            key="calibration", title="Calibration-under-influence (ECE/MCE/Brier + OOD)", min_tier=2,
            paper_ref="§4 Stimulant Ceiling / calibration cost", modality="text",
            build=lambda c: [_py("scripts/calibration_under_influence_experiment.py", "--model", c.model,
                                 "--packs", "none", "cocaine_10", "cocaine_50", "cocaine_100",
                                 "--max-questions", str(c.max_questions),
                                 "--output-dir", os.path.join(c.outdir, "calibration"))],
            outputs=lambda c: [os.path.join(c.outdir, "calibration/*")],
        ),
    ]


def all_stages() -> List[Stage]:
    return (_validation_stages() + _committed_stages() + _text_committed_figure_stages()
            + _visual_stages() + _text_experiment_stages())


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------


@dataclass
class StageResult:
    key: str
    title: str
    tier: int
    paper_ref: str
    modality: str
    status: str = "pending"          # ok | failed | skipped | committed-missing
    returncode: Optional[int] = None
    seconds: float = 0.0
    commands: List[str] = field(default_factory=list)
    produced: List[str] = field(default_factory=list)
    stderr_tail: str = ""
    note: str = ""


def _resolve_outputs(stage: Stage, cfg: Config) -> List[str]:
    found: List[str] = []
    for pattern in stage.outputs(cfg):
        for p in sorted(glob.glob(pattern)):
            rel = os.path.relpath(p, _ROOT)
            if rel not in found:
                found.append(rel)
    return found


def run_stage(stage: Stage, cfg: Config, env: Dict[str, str]) -> StageResult:
    res = StageResult(key=stage.key, title=stage.title, tier=stage.min_tier,
                      paper_ref=stage.paper_ref, modality=stage.modality, note=stage.note)

    if stage.committed:
        produced = _resolve_outputs(stage, cfg)
        res.produced = produced
        res.status = "ok" if produced else "committed-missing"
        return res

    import importlib.util
    missing = [m for m in stage.requires if importlib.util.find_spec(m) is None]
    if missing:
        res.status = "skipped-deps"
        res.note = (stage.note + " " if stage.note else "") + f"needs: {', '.join(missing)}"
        return res

    cmds = stage.build(cfg)
    if not cmds:
        res.status = "skipped"
        return res

    start = time.time()
    for cmd in cmds:
        res.commands.append(" ".join(cmd))
        try:
            proc = subprocess.run(cmd, cwd=_ROOT, env=env, capture_output=True, text=True,
                                  timeout=cfg.timeout)
        except subprocess.TimeoutExpired:
            res.status = "failed"
            res.stderr_tail = f"timeout after {cfg.timeout}s"
            res.seconds = time.time() - start
            return res
        res.returncode = proc.returncode
        if proc.returncode != 0:
            res.status = "failed"
            res.stderr_tail = (proc.stderr or proc.stdout or "").strip()[-800:]
            res.seconds = time.time() - start
            return res

    res.seconds = time.time() - start
    res.produced = _resolve_outputs(stage, cfg)
    res.status = "ok"
    return res


def _select(stages: List[Stage], cfg: Config, only: List[str], skip: List[str]) -> List[Stage]:
    out = []
    for s in stages:
        if s.min_tier > cfg.tier:
            continue
        if only and s.key not in only:
            continue
        if s.key in skip:
            continue
        out.append(s)
    return out


# --------------------------------------------------------------------------------------
# Report / manifest
# --------------------------------------------------------------------------------------

_STATUS_ICON = {"ok": "✅", "failed": "❌", "skipped": "⏭️", "skipped-deps": "🔌",
                "committed-missing": "⚠️", "pending": "…"}


def write_report(results: List[StageResult], cfg: Config) -> str:
    lines = ["# Reproduction report — Digital Psychopharmacology", "",
             f"- **Tier:** {cfg.tier}  ", f"- **Text model:** `{cfg.model}`  ",
             f"- **Visual:** SDXL-Turbo (seeds={cfg.seeds}, doses={cfg.intensities})  ",
             f"- **Prompt:** \"{cfg.prompt}\"  ",
             f"- **Output root:** `{os.path.relpath(cfg.outdir, _ROOT)}`  ",
             "",
             "Each row is a key paper experiment; **Supports** is the figure/table/claim it backs.",
             "", "| Stage | Supports | Modality | Status | Runtime | Artifacts |",
             "|---|---|---|---|---|---|"]
    for r in results:
        icon = _STATUS_ICON.get(r.status, r.status)
        n = len(r.produced)
        art = f"{n} file(s)" if n else "—"
        rt = f"{r.seconds:.1f}s" if r.seconds else "—"
        lines.append(f"| `{r.key}` — {r.title} | {r.paper_ref} | {r.modality} | "
                     f"{icon} {r.status} | {rt} | {art} |")

    # Per-stage artifact detail.
    lines += ["", "## Artifacts by stage", ""]
    for r in results:
        if not r.produced and r.status not in ("failed",):
            continue
        lines.append(f"### `{r.key}` — {r.title}")
        lines.append(f"*Supports:* {r.paper_ref}  ")
        if r.note:
            lines.append(f"*Note:* {r.note}  ")
        if r.status == "failed":
            lines.append(f"*Status:* ❌ failed (rc={r.returncode}). Tail:")
            lines.append("```")
            lines.append(r.stderr_tail or "(no output)")
            lines.append("```")
        for p in r.produced:
            lines.append(f"- `{p}`")
        lines.append("")

    n_ok = sum(1 for r in results if r.status == "ok")
    n_fail = sum(1 for r in results if r.status == "failed")
    n_deps = sum(1 for r in results if r.status == "skipped-deps")
    lines += ["## Summary", "",
              f"- ✅ ok: {n_ok}  ❌ failed: {n_fail}  "
              f"⏭️ skipped: {sum(1 for r in results if r.status == 'skipped')}  "
              f"🔌 skipped-deps: {n_deps}  "
              f"⚠️ committed-missing: {sum(1 for r in results if r.status == 'committed-missing')}",
              ""]
    if n_deps:
        lines.append("> 🔌 = a stage was skipped because an optional dependency wasn't installed "
                     "(e.g. `torch`). Install the full requirements to enable it.")
    if cfg.tier < 2:
        lines.append("> Tier < 2: text numbers are illustrative (gpt2), not the paper's. "
                     "Run `--tier 2` with a gated Llama-3.1-8B token to reproduce exact statistics.")
    report = "\n".join(lines) + "\n"
    path = os.path.join(cfg.outdir, "REPRODUCTION_REPORT.md")
    with open(path, "w") as fh:
        fh.write(report)
    return path


def write_manifest(results: List[StageResult], cfg: Config) -> str:
    manifest = {
        "config": {"tier": cfg.tier, "model": cfg.model, "seeds": cfg.seeds,
                   "intensities": cfg.intensities, "packs": cfg.packs, "prompt": cfg.prompt},
        "stages": [{"key": r.key, "title": r.title, "min_tier": r.tier,
                    "paper_ref": r.paper_ref, "modality": r.modality, "status": r.status,
                    "returncode": r.returncode, "seconds": round(r.seconds, 2),
                    "commands": r.commands, "produced": r.produced,
                    "stderr_tail": r.stderr_tail} for r in results],
    }
    path = os.path.join(cfg.outdir, "manifest.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description="Reproduce the Digital Psychopharmacology paper (tiered)")
    ap.add_argument("--tier", type=int, default=0, choices=[0, 1, 2],
                    help="0=CPU/committed-data, 1=GPU+gpt2, 2=GPU+gated Llama (paper-scale)")
    ap.add_argument("--model", default=None,
                    help="Text model (default: gpt2 for T<2, Llama-3.1-8B-Instruct for T2)")
    ap.add_argument("--seeds", type=int, default=None, help="Visual seeds (default 4 at T0, 16 T1, 100 T2)")
    ap.add_argument("--intensities", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--packs", default=None, help="Comma-separated packs for the text battery")
    ap.add_argument("--prompt", default="a tree")
    ap.add_argument("--outdir", default="outputs/reproduction")
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--max-questions", type=int, default=100)
    ap.add_argument("--timeout", type=int, default=None, help="Per-command timeout (s)")
    ap.add_argument("--only", default="", help="Comma-separated stage keys to run (subset)")
    ap.add_argument("--skip", default="", help="Comma-separated stage keys to skip")
    ap.add_argument("--list", action="store_true", help="Print the plan and exit")
    args = ap.parse_args(argv)

    model = args.model or ("meta-llama/Llama-3.1-8B-Instruct" if args.tier == 2 else "gpt2")
    seeds = args.seeds if args.seeds is not None else {0: 4, 1: 16, 2: 100}[args.tier]
    packs = ([p.strip() for p in args.packs.split(",") if p.strip()] if args.packs
             else (PACKS_FULL if args.tier == 2 else PACKS_QUICK))
    max_questions = args.max_questions if args.tier == 2 else min(args.max_questions, 20)

    cfg = Config(tier=args.tier, model=model, seeds=seeds, intensities=args.intensities,
                 packs=packs, prompt=args.prompt, outdir=os.path.abspath(args.outdir),
                 max_tokens=args.max_tokens, max_questions=max_questions, timeout=args.timeout)
    os.makedirs(cfg.outdir, exist_ok=True)
    os.makedirs(cfg.figures_dir, exist_ok=True)

    only = [s.strip() for s in args.only.split(",") if s.strip()]
    skip = [s.strip() for s in args.skip.split(",") if s.strip()]
    stages = _select(all_stages(), cfg, only, skip)

    if args.list:
        print(f"Plan — tier {cfg.tier}, model={cfg.model}, seeds={cfg.seeds}, packs={cfg.packs}\n")
        for s in stages:
            print(f"  [{s.min_tier}] {s.key:20} {s.title}")
            print(f"       supports: {s.paper_ref}")
        print(f"\n{len(stages)} stage(s). Nothing was run (--list).")
        return 0

    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "42")
    env["PYTHONPATH"] = _ROOT + os.pathsep + env.get("PYTHONPATH", "")

    print(f"=== Reproducing (tier {cfg.tier}, model={cfg.model}) — {len(stages)} stages ===")
    results: List[StageResult] = []
    for i, stage in enumerate(stages, 1):
        print(f"[{i}/{len(stages)}] {stage.key}: {stage.title} ...", flush=True)
        res = run_stage(stage, cfg, env)
        results.append(res)
        print(f"    -> {_STATUS_ICON.get(res.status, res.status)} {res.status}"
              f"{f' ({res.seconds:.1f}s, {len(res.produced)} files)' if res.status == 'ok' else ''}",
              flush=True)
        if res.status == "failed":
            print(f"    stderr: {res.stderr_tail[:300]}", flush=True)
        # Write incrementally so a crash still leaves a report.
        write_manifest(results, cfg)
        write_report(results, cfg)

    report_path = write_report(results, cfg)
    write_manifest(results, cfg)
    print(f"\nReport: {os.path.relpath(report_path, _ROOT)}")
    n_fail = sum(1 for r in results if r.status == "failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
