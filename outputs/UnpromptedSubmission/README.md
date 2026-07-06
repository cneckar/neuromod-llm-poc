# Unprompted.au Submission

Working folder for the Unprompted.au (ILUMINA) conference submission of
*"Biomimetic Neuromodulation: Reversible Inference-Time Control and Spectral Safety
Auditing for LLMs."*

Drop everything for the submission here — the current paper source (`.tex` / `.bib`),
figures, and the dose-response result artifacts — and it can be edited in place.

Suggested layout (create as needed):

- `paper/` — the LaTeX source as it currently stands (main `.tex`, `.bib`, `.sty`).
- `figures/` — figures referenced by the paper, including the N=100 dose-response
  panels (`dose_response_curves.png`, `specter_control.png`, `ood_capacity.png`,
  `safety_boundary_trigger_rates.png`).
- `results/` — the N=100 analysis artifacts (`sdxl_turbo_n100*.csv`,
  `trend_summary.csv`, `dose_curves.csv`) plus the OOD / safety-boundary CSVs.

Modality note (so claims stay correct): the N=100 dose-response study is the **image**
model (SDXL-Turbo diffusion). The **"Stimulant Ceiling" / flatline** is a **text**
result on RLHF-tuned **instruct** LLMs — a different model and modality. Keep the two
separate when integrating results.
