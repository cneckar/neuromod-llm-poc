# Running neuromod on real GPUs

Two independent workloads. Do **Path A** first — it's the cheap, fast one that produces the
conference data and needs nothing but a Google account.

- **Path A — Research pilot** (SDXL-Turbo image dose-response). Open weights, no tokens. Colab
  or any GPU box. Minutes of compute. → the decision matrix + real dose curves.
- **Path B — Live chat demo** (RunPod Serverless + Cloudflare). Scale-to-zero LLM inference
  streamed to the browser. Needs a RunPod account + API key, a container registry, a Cloudflare
  account, and an HF token for gpt-oss.

---

## Path A — Research pilot (fastest)

### A1. Colab (recommended, zero setup)

Open `notebooks/run_pilot_colab.ipynb` in Google Colab
(`File -> Open notebook -> GitHub -> cneckar/neuromod-llm-poc`), set
`Runtime -> Change runtime type -> GPU` (a free **T4** works), and **Run all**. It clones the
branch, installs deps, runs the pilot, prints the decision matrix, renders curves, and builds
the vitals GIF. Last cell zips + downloads the outputs.

### A2. Any GPU box / RunPod Pod (equivalent, CLI)

```bash
git clone https://github.com/cneckar/neuromod-llm-poc.git && cd neuromod-llm-poc
git checkout claude/relaxed-johnson-mqepm4          # or 'main' once PRs are merged
pip install -e . open_clip_torch lpips              # torch/diffusers pulled by the package
# Pilot: all four threads, N=16, coarse grid
python scripts/run_pilot.py --model sdxl-turbo --seeds 16 \
    --intensities 0.0,0.25,0.5,0.75,1.0 --outdir outputs/pilot
cat outputs/pilot/decision_matrix.csv               # the ranked headline pick
```

### A3. Scale to the full study (#12)

```bash
python scripts/run_pilot.py --model sdxl-turbo --seeds 100 \
    --intensities 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 --outdir outputs/full
# Resumable: safe to stop/restart; already-done (pack,intensity,seed) rows are skipped.
```

> **Known gap (#21):** thread B (safety boundary) shows `NaN` until the `SafetyOracle` +
> re-enabled SD checker are wired into the runner. Threads A/C/D produce real signals now.

---

## Path B — Live chat demo (RunPod Serverless + Cloudflare)

**Prerequisites:** RunPod account + API key; a container registry (Docker Hub / GHCR); Cloudflare
account with `wrangler`; a Hugging Face token (gpt-oss is open but gated behind an accept).

### B1. Build & push the D0 image

```bash
# from repo root
docker build -f deploy/runpod/Dockerfile -t <registry>/neuromod-runpod:latest .
docker push <registry>/neuromod-runpod:latest
```

### B2. Cache the weights on a network volume

Create a RunPod **Network Volume**, attach it to a throwaway Pod, and pre-download so cold
starts don't refetch:

```bash
export HF_HOME=/runpod-volume/hf
huggingface-cli login          # paste HF token
huggingface-cli download openai/gpt-oss-20b
```

### B3. Create the Serverless endpoint

In the RunPod console → Serverless → New Endpoint:
- Container image: `<registry>/neuromod-runpod:latest`
- **Min workers: 0** (scale-to-zero), Max workers: 2–3, FlashBoot: on
- Attach the network volume from B2
- Env: `MODEL_NAME=openai/gpt-oss-20b`, `HF_HOME=/runpod-volume/hf`,
  `HUGGINGFACE_TOKEN=<secret>`, `STREAMING=1`
- (Reference values live in `deploy/runpod/runpod.endpoint.example.json`.)

### B4. Smoke-test the endpoint

```bash
curl -s -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input":{"messages":[{"role":"user","content":"Describe a sunset."}],
                "pack_name":"lsd","intensity":0.7,"max_tokens":128}}'
```

You should get back a neuromodulated completion. Confirm the endpoint scales back to 0 workers
when idle (RunPod dashboard) → you're only billed for those seconds.

### B5. Deploy the Cloudflare Worker

```bash
cd deploy/cloudflare
npm install
# edit wrangler.toml -> RUNPOD_ENDPOINT_ID = <ENDPOINT_ID>
wrangler secret put RUNPOD_API_KEY      # paste the RunPod key (stays server-side)
wrangler secret put API_KEY             # optional client auth; omit for open dev
wrangler deploy
```

### B6. Use it

Open the deployed Worker URL → the chat UI streams tokens (pick a pack + intensity). The
browser talks only to the Worker; the RunPod key never leaves the edge.

---

## Cost notes

- **Path A:** SDXL-Turbo ~0.5–1s/image on a T4; the N=16 pilot is minutes; Colab T4 is free.
- **Path B:** per-second billing only during inference; idle = \$0. A ~500-token stream is a
  few GPU-seconds. Cap spend with `workersMax` and the idle timeout (D4, #19).

## Where things are tracked

- Remaining GPU acceptance criteria: **#21**.
- Full run → figures → paper: **#12**, **#13**.
- Deployment follow-ons: D2 **#17**, D3 (FLUX) **#18**, D4 (cost guardrails) **#19**.
