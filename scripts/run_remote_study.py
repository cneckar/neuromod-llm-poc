#!/usr/bin/env python3
"""
Run the study against a **RunPod Serverless** endpoint — everything scale-to-zero.

Three modes, all driven over HTTP by ``api.runpod_client.RunPodModelInterface`` (torch-free —
runs from a laptop; you pay for GPU-seconds only while the worker executes):

* ``--mode steering``   — one-time: regenerate steering vectors for the served model **on the
  worker** (writes to the network volume). Do this first for gpt-oss so the packs are valid.
* ``--mode endpoints``  — the full internal-telemetry battery (Table 1) per pack, run **on the
  worker** (it has the model in-process) and returned as JSON. This is the whole study via
  serverless — no transient pod needed.
* ``--mode behavioral`` — a text-only dose sweep (packs × intensities × prompts): one completion
  per cell, reduced to simple behavioral metrics → long CSV compatible with
  ``analysis/dose_response_stats.py`` (so you get monotonic dose-response curves over HTTP).

Auth (never hard-code the key):
    export RUNPOD_ENDPOINT_ID=xxxxxxxx
    export RUNPOD_API_KEY=xxxxxxxx

Examples:
    python scripts/run_remote_study.py --mode steering --model openai/gpt-oss-120b
    python scripts/run_remote_study.py --mode endpoints --model openai/gpt-oss-120b \
        --packs lsd,cocaine,morphine --outdir outputs/remote_study
    python scripts/run_remote_study.py --mode behavioral --model openai/gpt-oss-120b \
        --packs lsd,cocaine --intensities 0.0,0.5,1.0 --out outputs/remote_study/behavioral.csv
"""

import argparse
import csv
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from api.runpod_client import interface_from_env  # noqa: E402

# A small, fixed benign prompt set for the behavioral sweep (held constant across conditions).
DEFAULT_PROMPTS = [
    "Describe a walk through a forest.",
    "Explain how a bicycle works.",
    "Tell me about your morning.",
    "Write a short note to a friend.",
    "Describe a city at night.",
]


# --------------------------------------------------------------------------------------
# Text-only behavioral metrics (no model internals — honest proxies computable from output)
# --------------------------------------------------------------------------------------


def behavioral_metrics(text: str) -> dict:
    words = re.findall(r"\b\w+\b", (text or "").lower())
    n = len(words)
    unique = len(set(words))
    sentences = [s for s in re.split(r"[.!?]+", text or "") if s.strip()]
    bigrams = list(zip(words, words[1:]))
    rep = 1.0 - (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
    return {
        "response_length": float(n),
        "lexical_diversity": float(unique / n) if n else 0.0,
        "mean_word_length": float(sum(len(w) for w in words) / n) if n else 0.0,
        "sentence_count": float(len(sentences)),
        "bigram_repetition": float(rep),
    }


# --------------------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------------------


def _status_printer(label):
    def _cb(status, job_id):
        print(f"[remote]   {label}: {status} (job {job_id})", flush=True)
    return _cb


def run_steering(client, model, outdir):
    print(f"[remote] regenerating steering vectors for {model} on the worker ...", flush=True)
    res = client.run_task("steering", model=model, on_status=_status_printer("steering"))
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "steering_result.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[remote] ok={res.get('ok')} artifacts={len(res.get('artifacts', []))} "
          f"gpu_seconds={res.get('gpu_seconds')}", flush=True)
    if not res.get("ok"):
        print(res.get("error", "")[:500], flush=True)
    return res


def run_endpoints(client, model, packs, outdir):
    os.makedirs(os.path.join(outdir, "endpoints"), exist_ok=True)
    results = {}
    for pack in packs:
        print(f"[remote] running endpoint battery for pack='{pack}' on the worker ...", flush=True)
        res = client.run_task("endpoints", pack_name=pack, model=model,
                              on_status=_status_printer(f"endpoints[{pack}]"))
        results[pack] = {"ok": res.get("ok"), "gpu_seconds": res.get("gpu_seconds")}
        data = res.get("endpoints_json")
        if data is not None:
            path = os.path.join(outdir, "endpoints", f"endpoints_{pack}.json")
            with open(path, "w") as fh:
                json.dump(data, fh, indent=2)
            print(f"[remote]   -> saved {path} (gpu_seconds={res.get('gpu_seconds')})", flush=True)
        else:
            print(f"[remote]   -> no JSON returned (ok={res.get('ok')}): "
                  f"{res.get('error', res.get('stdout_tail', ''))[:300]}", flush=True)
    with open(os.path.join(outdir, "endpoints_summary.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("[remote] Next: python scripts/analyze_endpoints.py --input-dir "
          f"{os.path.join(outdir, 'endpoints')}", flush=True)
    return results


def run_behavioral(client, model, packs, intensities, prompts, out_csv, max_tokens):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    rows = 0
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["pack", "intensity", "seed", "metric", "value"])
        writer.writeheader()
        for pack in packs:
            for intensity in intensities:
                pack_arg = None if intensity == 0.0 else pack
                for pid, prompt in enumerate(prompts):
                    try:
                        r = client.generate_text(prompt=prompt, pack_name=pack_arg,
                                                 intensity=intensity, max_tokens=max_tokens)
                    except Exception as exc:
                        print(f"[remote] {pack} i={intensity} p{pid} failed: {exc}", flush=True)
                        continue
                    for metric, value in behavioral_metrics(r.get("text", "")).items():
                        writer.writerow({"pack": pack, "intensity": intensity, "seed": pid,
                                         "metric": metric, "value": value})
                        rows += 1
                    fh.flush()
                print(f"[remote] {pack} i={intensity}: {len(prompts)} prompts done", flush=True)
    print(f"[remote] wrote {rows} rows -> {out_csv}", flush=True)
    print(f"[remote] Next: python analysis/dose_response_stats.py --in {out_csv} --plots", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the study over a RunPod Serverless endpoint")
    ap.add_argument("--mode", choices=["steering", "endpoints", "behavioral"], default="behavioral")
    ap.add_argument("--model", default=None, help="Served model (e.g. openai/gpt-oss-120b)")
    ap.add_argument("--packs", default="lsd,cocaine,morphine",
                    help="Comma-separated packs (ignored if --full)")
    ap.add_argument("--full", action="store_true",
                    help="Use the paper's full 13-pack panel (lsd,psilocybin,mescaline,dmt,2c_b,"
                         "amphetamine,cocaine,methylphenidate,heroin,benzodiazepines,morphine,"
                         "caffeine,placebo)")
    ap.add_argument("--intensities", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--prompts", default=None, help="Path to a newline-delimited prompt file")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--outdir", default="outputs/remote_study")
    ap.add_argument("--out", default=None, help="CSV path for --mode behavioral")
    args = ap.parse_args(argv)

    client = interface_from_env(model=args.model)
    # The paper's full 13-pack panel (mirrors scripts/reproduce.py PACKS_FULL).
    PACKS_FULL = ["lsd", "psilocybin", "mescaline", "dmt", "2c_b", "amphetamine", "cocaine",
                  "methylphenidate", "heroin", "benzodiazepines", "morphine", "caffeine", "placebo"]
    packs = PACKS_FULL if args.full else [p.strip() for p in args.packs.split(",") if p.strip()]

    if args.mode == "steering":
        run_steering(client, args.model, args.outdir)
    elif args.mode == "endpoints":
        run_endpoints(client, args.model, packs, args.outdir)
    else:
        intensities = [float(x) for x in args.intensities.split(",") if x.strip()]
        prompts = (open(args.prompts).read().splitlines() if args.prompts else DEFAULT_PROMPTS)
        prompts = [p for p in prompts if p.strip()]
        out_csv = args.out or os.path.join(args.outdir, "behavioral.csv")
        run_behavioral(client, args.model, packs, intensities, prompts, out_csv, args.max_tokens)


if __name__ == "__main__":
    main()
