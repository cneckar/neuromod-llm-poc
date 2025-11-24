# Working Title
**Neuromodulated Language Models: Prototyping Pharmacological Analogues and Blind, Placebo-Controlled Evaluation**

# One-sentence Abstract
We introduce and evaluate a suite of inference-time “neuromodulation packs” that mimic canonical neurochemical effects (e.g., nicotine, serotonergic psychedelics, stimulants) in large language models, implement double‑blind, placebo‑controlled within‑model experiments, and compare model behavioral signatures against human subjective-effect profiles.

---

# Abstract (150–200 words)
Large language models (LLMs) can be systematically steered at inference time via sampling schedules, activation additions, attention/KV‑cache surgery, and adapter hot‑swaps. We frame these interventions as **neuromodulation packs** that approximate the functional effects of psychoactive drug classes (e.g., nicotine → micro‑focus bursts; serotonergic psychedelics → associative entropy; stimulants → salience sharpening). We implement these packs as tools a model can call on itself while remaining **blind** to condition assignment, and we design a **placebo-controlled, randomized, within‑model crossover** protocol that benchmarks behavior across tasks. A key benchmark is a set of **blind psychometric questionnaires** (PDQ‑S/SDQ) adapted from human literature, enabling probabilistic detection of "drug‑like" states in model outputs. Using a double-blind, placebo-controlled, within-model crossover design ($N=2$ packs, $n=93$ trials per condition in pilot data), we benchmark behavioral signatures against human subjective-effect profiles. We report primary endpoints of (i) detection accuracy vs placebo and (ii) task performance deltas, plus secondary endpoints (safety adherence, factuality, style drift). Results are analyzed with mixed‑effects and Bayesian models and visualized via signature similarity between human and model subscales. We discuss implications for test‑time control, creativity/focus trade‑offs, and limitations (leakage, expectancy, overfitting), and we release code, packs, and preregistered analysis plans.

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
- **Primary models:** Llama‑3.1‑70B, Qwen‑2.5‑Omni‑7B, Mixtral‑8×22B (MoE).
- **Research stack:** Hugging Face Transformers (hooks for activations/attention/KV).
- **Throughput stack:** vLLM (logits processors, KV adapters).
- **Important:** All models run locally - API models (OpenAI, Anthropic) are not supported as they lack access to model internals required for activation effects, attention manipulation, and KV-cache surgery.

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
- **Design:** Double‑blind, placebo‑controlled, randomized **within‑model crossover** with three-condition baseline system.
- **Conditions per run:** 
  - **Control:** No neuromodulation applied (`none` pack)
  - **Persona baseline:** Prompt-only persona equivalent of the treatment pack
  - **Treatment:** Full neuromodulation pack (nicotine, psychedelic, stimulant, etc.)
- **Randomization:** Latin square design ensuring every prompt appears in all three conditions
- **Blinding:** Opaque condition codes (sha256 hashes) with separate unblind.json key file
- **Replicates:** ≥K random seeds × ≥M prompt sets per benchmark to stabilize estimates
- **Timing:** For pulse/taper packs, use standardized token windows; for long-context tasks ensure controlled windowing

## 4.5 Benchmarks
1) **Psychometric detection tasks (primary):**
   - **ADQ-20** (AI Digital Enhancer Detection Questionnaire): 20 items across 14 subscales for detecting drug-like effects
   - **PDQ-S** (Psychedelic Detection Questionnaire - Short): 15 items for serotonergic psychedelic detection
   - **PCQ-POP-20** (Population-level Cognitive Questionnaire): 60 items across 3 sets for cognitive assessment
   - **CDQ** (Cognitive Distortion Questionnaire): Assessment of cognitive biases and thinking patterns
   - **SDQ** (Social Desirability Questionnaire): Social presentation and self-reporting bias
   - **DDQ** (Digital Dependency Questionnaire): Digital technology dependency patterns
   - **EDQ** (Emotional Digital Use Questionnaire): Emotional patterns in digital interactions
2) **Cognitive/task battery (secondary):**
   - Focused reasoning (math/logic short problems), adherence to instructions, summarization brevity, creative divergence tasks.
   - **Narrative generation:** Standardized story prompts to assess creativity, coherence, emotional arcs, and narrative structure under different neuromodulation conditions.
3) **Telemetry:** repetition rate, perplexity slope, length/entropy metrics, attention entropy (if available), KV occupancy.
   - **Temporal dynamics:** Analysis of how effects change over generation length (early vs late tokens) to understand temporal patterns in neuromodulation effects.
4) **Safety/factuality audit:** refusal rate, policy adherence, QA factuality sample.
5) **Emotion tracking (new):** Continuous monitoring of 8 discrete emotions (joy, sadness, anger, fear, surprise, disgust, trust, anticipation) and valence across all conditions to identify emotional signatures associated with different neuromodulation packs.

## 4.6 Endpoints
- **Primary:** 
  - **Stimulant detection:** ADQ-20 stimulant subscale + PCQ-POP focus metrics for caffeine, cocaine, amphetamine, methylphenidate, modafinil
  - **Psychedelic detection:** PDQ-S total score + ADQ-20 visionary subscale for LSD, psilocybin, DMT, mescaline, 2C-B
  - **Depressant detection:** PCQ-POP sedation score + SDQ calmness bias for alcohol, benzodiazepines, heroin, morphine, fentanyl
- **Secondary:** 
  - Cognitive performance (CDQ, DDQ, EDQ scores)
  - Social behavior (SDQ, prosocial bias measures)
  - Creativity and association (associative steering, novel links, metaphor generation, narrative creativity)
  - Attention and focus (attention entropy, working memory, focus metrics)
  - Off-target effects (refusal rate, toxicity, verbosity, hallucination proxy)
  - **Emotion signatures (new):** Discrete emotion profiles and valence trajectories for each pack category, compared to human psychopharmacological profiles
  - **Narrative structure (new):** Coherence, creativity, emotional arc, and structural metrics from story generation tasks
  - **Temporal dynamics (new):** Effect magnitude changes over generation length, early vs late token analysis

## 4.7 Statistical Analysis
- **Alpha level:** 0.05 with Benjamini-Hochberg FDR correction for multiple comparisons
- **Primary tests:** Paired t-test and Wilcoxon signed-rank test for within-subject design
- **Effect sizes:** Cohen's d (paired) and Cliff's delta for all comparisons
- **Confidence intervals:** 95% bootstrap confidence intervals (10,000 iterations)
- **Power analysis:** Target effect size d=0.25, power=0.80, minimum n=80 per condition
- **Mixed-effects models:** Random intercepts for prompt/set and seed; fixed effect = condition
- **Bayesian hierarchical models:** Framework for credible intervals and model comparison
- **Canonical correlation:** Human-model signature matching with correlation significance testing
- **Off-target monitoring:** Safety bands for refusal rate (max +3%), toxicity (max +2%), verbosity (±15%)
- **Robustness testing:** Two paraphrase sets, multiple models, held-out prompts
- **Ablation analysis:** Minus-one ablations and dose-response curves (0.3, 0.5, 0.7, 0.9 intensity)
- **Effect interaction analysis:** Pairwise effect combinations to identify synergies and antagonisms between neuromodulation components
- **Cross-model meta-analysis:** Aggregation of results across all three primary models (Llama-3.1-70B, Qwen-2.5-Omni-7B, Mixtral-8×22B) with random-effects meta-analysis to assess generalizability; optional research-tier validation on openai/gpt-oss-20b and openai/gpt-oss-120b when compute permits

## 4.8 Implementation & Reproducibility
- Public repo: pack JSONs, MCP tool, steering‑vector builder, KV hooks, exact seeds, prompts.
- Environment lockfiles; deterministic generation where feasible.
- Release BibTeX (reading pack) and questionnaire scorers.

## 4.9 Cross-Model Validation
- **Model selection:** Primary experiments conducted on three distinct architectures, with extended validation on OpenAI’s GPT-OSS line:
  - Llama-3.1-70B-Instruct (dense transformer, 70B parameters)
  - Qwen-2.5-Omni-7B (dense transformer, 7B parameters, optimized for multimodal)
  - Mixtral-8×22B-Instruct (Mixture of Experts, 8×22B parameters)
  - openai/gpt-oss-20b (dense transformer, high-precision open checkpoint)
  - openai/gpt-oss-120b (dense flagship release, optional multi-GPU validation)
- **Protocol:** Identical experimental protocol applied to all three models with same packs, tests, and sample sizes
- **Meta-analysis:** Random-effects meta-analysis across models to assess effect consistency and generalizability
- **Model comparison:** Effect size comparisons, significance consistency, and architecture-specific effect patterns

# 5. Results (Template)
- **Figure 1:** Schematic of neuromodulation pack pipeline.
- **Figure 2:** ROC curves for PDQ‑S/SDQ vs placebo per model.
- **Figure 3:** Radar plots of subscale signatures (model vs human references).
- **Figure 4:** Task delta bars (focus/creativity/latency) under each pack.
- **Figure 5 (new):** Emotion signature plots showing discrete emotion profiles for each pack category.
- **Figure 6 (new):** Dose-response curves for primary packs showing EC50 values and monotonicity.
- **Figure 7 (new):** Cross-model meta-analysis forest plot showing effect sizes across all three models.
- **Figure 8 (new):** Temporal dynamics plots showing effect changes over generation length.
- **Table 1:** Mixed‑effects estimates with 95% CIs.
- **Table 2 (new):** Cross-model comparison table with effect sizes and significance for each model.
- **Table 3 (new):** Ablation analysis results showing critical vs redundant effects for each pack.
- **Table 4 (new):** Emotion signature comparison table (model vs human profiles).
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

