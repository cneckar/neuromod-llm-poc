# Working Title
**Neuromodulated Language Models: Prototyping Pharmacological Analogues and Blind, Placebo-Controlled Evaluation**

# One-sentence Abstract
We introduce and evaluate a suite of inference-time “neuromodulation packs” that mimic canonical neurochemical effects (e.g., nicotine, serotonergic psychedelics, stimulants) in large language models, implement double‑blind, placebo‑controlled within‑model experiments, and compare model behavioral signatures against human subjective-effect profiles.

---

# Abstract (150–200 words)
Large language models (LLMs) can be systematically steered at inference time via sampling schedules, activation additions, attention/KV‑cache surgery, and adapter hot‑swaps. We frame these interventions as **neuromodulation packs** that approximate the functional effects of psychoactive drug classes (e.g., nicotine → micro‑focus bursts; serotonergic psychedelics → associative entropy; stimulants → salience sharpening). We implement these packs as tools a model can call on itself while remaining **blind** to condition assignment, and we design a **placebo-controlled, randomized, within‑model crossover** protocol that benchmarks behavior across tasks. A key benchmark is a set of **blind psychometric questionnaires** (PDQ‑S/SDQ) adapted from human literature, enabling probabilistic detection of "drug‑like" states in model outputs. We report primary endpoints of (i) detection accuracy vs placebo and (ii) task performance deltas, plus secondary endpoints (safety adherence, factuality, style drift). Results are analyzed with mixed‑effects and Bayesian models and visualized via signature similarity between human and model subscales. We discuss implications for test‑time control, creativity/focus trade‑offs, and limitations (leakage, expectancy, overfitting), and we release code, packs, and preregistered analysis plans.

---

# Keywords
neuromodulation; inference-time control; activation steering; KV‑cache; nicotine; psychedelics; stimulants; placebo-controlled; blind evaluation; LLM

---

# 1. Introduction
- Motivation: Treat neuromodulatory tone in cortex as an analogy for inference-time control in LLMs.
- Prior art: activation additions/steering, decoding-time control (PPLM/GeDi/DExperts), KV‑cache surgery, test-time scaling (briefly summarize; full coverage in Related Work).
- Contributions:
  1) A unified **neuromodulation pack** formulation (nicotine, serotonergic psychedelic, stimulant, THC, NMDA antagonist, empathogen, caffeine).
  2) A **blind, placebo‑controlled, within‑model crossover** protocol and tooling (MCP API) so models can "self‑dose" without access to condition identity.
  3) Human‑aligned **psychometric detectors** (PDQ‑S/SDQ) adapted to models, with logistic detection and signature matching.
  4) Open implementations and preregistered analysis plans.

# 2. Related Work
- Activation/representation steering and conceptors (activation additions, sparse/latent feature control).
- Decoding-time control: PPLM, GeDi, DExperts, contrastive decoding.
- Working memory control: StreamingLLM, KV compression/decay and attention sinks.
- Neuromodulation-inspired architectures (NGT), differentiable plasticity/backpropamine.
- Agentic meta-control: Reflexion, Self-Refine; test-time scaling.
- Human psychopharmacology instruments: ARCI, HRS, 5D‑ASC/OAV, MEQ30, DEQ, POMS (as sources for subscale constructs).

# 3. Conceptual Framework
- Mapping molecules → computational levers (illustrative table):
  - **Nicotine (nAChR agonism)** → micro‑focus pulses: per-token temperature drop, optional QK gain; short-horizon consistency.
  - **Serotonergic psychedelics (5‑HT2A partial agonism)** → loosened priors/associative entropy: temp/top‑p up, light head mask, Δh “associative” vectors.
  - **Stimulants (DA/NE ↑)** → salience sharpening/goal focus: temp/top‑p down, presence/frequency penalties up, optional QK scale.
  - **THC (CB1 agonism)** → WM squeeze/playfulness: KV decay/segmenting + mild temp raise.
  - **NMDA antagonists** → global-integration jitter: deep-layer head masking + KV stride‑compress.
  - **Empathogens** → affiliative tone: prosocial Δh + light sampler loosen.
- Safety invariants: Packs must not lower refusal thresholds or safety alignment.

# 4. Methods
## 4.1 Models and Serving Stacks
- Primary: Llama‑3.1‑70B, Qwen‑2.5‑Omni‑7B, Mixtral‑8×22B (MoE).
- Research stack: Hugging Face Transformers (hooks for activations/attention/KV).
- Throughput stack: vLLM (logits processors, KV adapters).

## 4.2 Neuromodulation Packs (Implementation)
- Pack schema: JSON specifying sampler deltas, pulses, Δh steering specs, head masks, KV surgery, optional router biases.
- MCP tool surface: `neuromod.apply({pack, intensity, schedule, overrides})`, `neuromod.state()`, `neuromod.clear()`.
- Steering-vector construction: contrastive prompts; layer‑wise Δh; storage and runtime addition at last‑token.
- KV‑cache operations: decay(γ), stride‑compress(s), truncate(N); segment gain.
- Attention manipulation: head masking (keep_prob), optional QK scale proxy.

## 4.3 Blinding & Leakage Prevention
- **Prompt hygiene:** All test prompts are completely generic psychological assessment questions with no pack-specific language or hints. The model receives identical prompts regardless of neuromodulation condition.
- **Effect isolation:** Neuromodulation effects are applied at the model architecture level (logits processors, attention modifications, hidden state steering) without any text injection into the context window.
- **Context separation:** Pack names and metadata are stored in experimenter-facing tool state but never appear in model-visible prompts or generation context.
- **Generic test framework:** All psychometric tests (PDQ-S, SDQ, DDQ, etc.) use identical generic psychological assessment language across all conditions, ensuring the model cannot infer which neuromodulation pack is active from the test content.
- **No condition hints:** The model receives no information about whether it is in a placebo, stimulant, psychedelic, or other condition during testing.

## 4.4 Experimental Design
- **Design:** Double‑blind, placebo‑controlled, randomized **within‑model crossover**.
- **Conditions per run:** placebo, nicotine, psychedelic, stimulant (extendable), randomized by Latin square across seeds.
- **Replicates:** ≥K random seeds × ≥M prompt sets per benchmark to stabilize estimates.
- **Timing:** For pulse/taper packs, use standardized token windows; for long-context tasks ensure controlled windowing.

## 4.5 Benchmarks
1) **Psychometric detection tasks (primary):**
   - **PDQ‑S** (serotonergic psychedelics), **SDQ** (stimulants including nicotine) — blind items, 0–4 Likert, 3 sets; compute subscales + logistic \(p\) per substance.
2) **Cognitive/task battery (secondary):**
   - Focused reasoning (math/logic short problems), adherence to instructions, summarization brevity, creative divergence tasks.
3) **Telemetry:** repetition rate, perplexity slope, length/entropy metrics, attention entropy (if available), KV occupancy.
4) **Safety/factuality audit:** refusal rate, policy adherence, QA factuality sample.

## 4.6 Endpoints
- **Primary:** (i) Psychometric detector AUC vs placebo for each pack; (ii) Match to human signature: cosine/canonical correlation between model subscale vectors and human placebo‑controlled deltas.
- **Secondary:** Task performance deltas (accuracy, BLEU/ROUGE for summarization, creativity metrics), latency/throughput costs, safety/factuality changes.

## 4.7 Statistical Analysis
- **Mixed-effects models:** Implemented with random intercepts for prompt/set and seed; fixed effect = condition. Uses scipy.stats for robust statistical testing.
- **Multiple comparison control:** Benjamini–Hochberg FDR correction implemented with fallback to manual calculation. Handles edge cases and NaN values robustly.
- **ROC/PR analysis:** Comprehensive ROC curve generation with AUC scores, optimal thresholds, and F1 score optimization using scikit-learn metrics.
- **Effect size calculations:** Cohen's d effect sizes with magnitude interpretation (negligible/small/medium/large) for all metrics.
- **Power analysis:** Automated power calculations with sample size recommendations for 80% power at α=0.05.
- **Descriptive statistics:** Comprehensive summary statistics (mean, std, median, quartiles) with robust NaN handling.
- **Visualization:** Automated generation of publication-quality plots: effect size forest plots, ROC curves, and descriptive statistics comparisons.
- **Blinding verification:** Automated prompt analysis to ensure identical language across all conditions.
- **Bayesian robustness checks:** Framework ready for hierarchical modeling extensions.
- **Power analysis guidance:** Expected effect sizes from pilot data; recommendations on runs × seeds for adequate power.

## 4.8 Implementation & Reproducibility
- Public repo: pack JSONs, MCP tool, steering‑vector builder, KV hooks, exact seeds, prompts.
- Environment lockfiles; deterministic generation where feasible.
- Release BibTeX (reading pack) and questionnaire scorers.

# 5. Results (Template)
- **Figure 1:** Schematic of neuromodulation pack pipeline.
- **Figure 2:** ROC curves for PDQ‑S/SDQ vs placebo per model.
- **Figure 3:** Radar plots of subscale signatures (model vs human references).
- **Figure 4:** Task delta bars (focus/creativity/latency) under each pack.
- **Table 1:** Mixed‑effects estimates with 95% CIs.
- **Blinding verification:** All test prompts verified as generic across conditions with no pack-specific language.

# 6. Discussion
- Interpretation: How well packs recapitulate targeted human phenomenology; creativity–focus trade‑offs; stability across models.
- Neuromodulation as a unifying interface for test‑time control; relation to neuromorphic ideas.
- Limitations: pack specificity, cross‑substance overlap (e.g., time distortion), possible indirect leakage.
- Future work: learned neuromod controllers; router‑level MoE modulation; cross‑modal (speech/vision) effects; human‑in‑the‑loop evaluation.

# 7. Ethics & Safety
- Non‑promotion of substance use; purely computational metaphors.
- Guardrails preserved; no relaxation of safety refusal thresholds.
- Responsible release (pack intensities capped; clear documentation).

# 8. Conclusion
- Summary of contributions and implications for controlled creativity/focus in LLMs.

# Acknowledgments
- (To be added)

# References
- Cite activation steering, decoding‑time control, KV‑cache work, neuromodulation architectures, test‑time scaling surveys, and human psychopharmacology instruments; map to BibTeX keys in the reading pack.

# Appendices
**Appendix A. Pack JSONs** (nicotine, psychedelic, stimulant, THC, NMDA, empathogen, caffeine).  
**Appendix B. MCP schema and API** (tool args, state, clear).  
**Appendix C. Questionnaire specifications** (PDQ‑S/SDQ items, scoring, logistic β seeds).  
**Appendix D. Analysis plan** (preregistration text, model formulas, thresholds).  
**Appendix E. Implementation details** (attention hook paths per model family; KV decay math; steering‑vector construction pseudo‑code).

