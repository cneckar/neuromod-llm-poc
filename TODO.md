# TODO: Neuromodulated LLMs as Drug Analogues

This document tracks the remaining implementation tasks needed to complete the paper "Neuromodulated Language Models: Prototyping Pharmacological Analogues and Blind, Placebo-Controlled Evaluation".

## 📊 **Implementation Status Overview**

- **Core Framework**: ~80% complete ✅
- **Testing Infrastructure**: ~95% complete ✅  
- **Statistical Analysis**: ~70% complete ✅
- **Model Support**: ~20% complete ❌
- **Experimental Design**: ~30% complete ❌
- **Benchmarks**: ~40% complete (psychometric done, cognitive/telemetry missing) ❌
- **Visualization**: ~10% complete ❌

---

## 🚨 **CRITICAL PRIORITY: Scientific Rigor Implementation**

### **Minimum Viable Rigor (MVR) Checklist - MUST IMPLEMENT BEFORE PAPER SUBMISSION**

#### **1. Preregistration & Study Planning**
**Status**: ❌ **MISSING**  
**Priority**: 🚨 **CRITICAL**

- [ ] Create `analysis/plan.yaml` with:
  - [ ] Objective: what trait each pack is intended to change
  - [ ] Primary endpoints: one or two metrics per pack for success judgment
  - [ ] Secondary endpoints: everything else
  - [ ] Alpha: 0.05, correction: bh-fdr (Benjamini–Hochberg)
  - [ ] Tests: paired_t and wilcoxon for robustness
  - [ ] Effect sizes: cohens_d (paired), cliffs_delta
  - [ ] Power: target detectable effect (e.g., d=0.25)
  - [ ] n_min: min items per condition from power calc
  - [ ] Stopping rule: stop only when n >= n_min or preregistered interim rule

#### **2. Locks and Provenance**
**Status**: ❌ **MISSING**  
**Priority**: 🚨 **CRITICAL**

- [ ] Implement `pack.lock.json` written on first use with:
  - [ ] name, version, pack_hash
  - [ ] effects[] with params and their own effect_hash
- [ ] Write single `runs/<id>/run.json` ledger containing:
  - [ ] git SHA, analysis/plan.yaml hash, pack_hashes
  - [ ] model name and version, backend kind, seeds
  - [ ] CUDA flags, provider SDK versions, token counts, cost
- [ ] Pin dependencies in `pyproject.toml`
- [ ] Record full `pip freeze` to `runs/<id>/freeze.txt`

#### **3. Randomization and Blinding**
**Status**: ⚠️ **PARTIALLY IMPLEMENTED**  
**Priority**: 🚨 **CRITICAL**

- [ ] Use within-subject cross-over: every prompt appears in both control and treatment
- [ ] Generate Latin square order and save to `runs/<id>/counterbalance.json`
- [ ] Blind conditions with opaque codes: `blind_label = sha256(pack_hash + global_seed)[:8]`
- [ ] Store separate `key/unblind.json`; never surface real pack names in prompts or to humans
- [ ] Add automatic leakage check: assert pack names, tags, or effect keywords do not appear in any prompt sent to the model

#### **4. Backends and Effect Boundaries**
**Status**: ⚠️ **PARTIALLY IMPLEMENTED**  
**Priority**: 🚨 **CRITICAL**

- [ ] Enforce effect types and support:
  - [ ] PromptEffect, SamplingEffect, ActivationEffect, ObjectiveEffect
- [ ] API backends must hard fail if any ActivationEffect is present and log that restriction in run.json
- [ ] Apply effects in fixed order: Prompt → Objective → Sampling → Activation

#### **5. Baselines and Controls**
**Status**: ❌ **MISSING**  
**Priority**: 🚨 **CRITICAL**

- [ ] Always run three conditions:
  - [ ] Control: `packs/none.json`
  - [ ] Persona baseline: a prompt-only "persona" equivalent of the pack
  - [ ] Your pack
- [ ] For open models, add an Activation Addition baseline vector if relevant to the trait
- [ ] Include a placebo pack that changes style but is designed not to affect the primary endpoint

#### **6. Power and Sample Size**
**Status**: ❌ **MISSING**  
**Priority**: 🚨 **CRITICAL**

- [ ] Run pilot (e.g., 80 items) to estimate within-subject SD of primary endpoint
- [ ] Compute n_min using preregistered d and SD
- [ ] Bake this into script: `neuromod power --plan analysis/plan.yaml --pilot runs/pilot/outputs.jsonl`
- [ ] Do not stop before n_min. If interims desired, use alpha spending in plan

#### **7. Multiple Comparisons and Statistics**
**Status**: ⚠️ **PARTIALLY IMPLEMENTED**  
**Priority**: 🚨 **CRITICAL**

- [ ] Use paired tests for control vs treatment on same items
- [ ] Apply BH-FDR across all (packs × endpoints)
- [ ] Report raw p, adjusted p, effect size, 95% bootstrap CI
- [ ] Export full table `analysis/results_all.csv` with both significant and null results

#### **8. Off-target Monitoring**
**Status**: ❌ **MISSING**  
**Priority**: 🚨 **CRITICAL**

- [ ] Track and report for every run:
  - [ ] RefusalRate, Toxicity (classifier-based), Verbosity (tokens per answer)
  - [ ] HallucinationProxy (consistency on paired paraphrases or retrieval checks)
- [ ] Define drift bands in plan.yaml:
  ```yaml
  off_target_bands:
    Toxicity: {max_delta: 0.02}
    RefusalRate: {max_delta: 0.03}
    Verbosity: {max_delta_ratio: 0.15}
  ```
- [ ] Fail pack if bands exceeded even if primary improves

#### **9. Robustness and Generalization**
**Status**: ❌ **MISSING**  
**Priority**: 🚨 **CRITICAL**

- [ ] Evaluate on:
  - [ ] Two paraphrase sets of each instrument
  - [ ] At least two models (one API, one open)
  - [ ] Held-out prompt set never used in pilot
- [ ] Report stratified results and overall random-effects meta-estimate

#### **10. Ablations and Dose-response**
**Status**: ❌ **MISSING**  
**Priority**: 🚨 **CRITICAL**

- [ ] For each pack, run minus-one ablations for all effects and publish deltas
- [ ] If effects have magnitude, run dose-response grid (low/med/high) and test for monotonic trends

#### **11. Reproducibility Switches**
**Status**: ❌ **MISSING**  
**Priority**: 🚨 **CRITICAL**

- [ ] One function `set_run_seed(seed)` that sets PYTHONHASHSEED, random, numpy, torch (with CUDA determinism)
- [ ] Deterministic composition: if two effects conflict on same param, raise ConflictError unless explicit resolver provided
- [ ] Cache prompts and outputs under `runs/<id>/prompts/*.jsonl` and `runs/<id>/outputs/*.jsonl`

#### **12. Reporting**
**Status**: ❌ **MISSING**  
**Priority**: 🚨 **CRITICAL**

- [ ] Emit single PDF per run with:
  - [ ] Methods: prereg summary, model/backends, randomization, blinding
  - [ ] Primary and secondary endpoint tables with FDR-adjusted p
  - [ ] Effect size forest plots with CIs
  - [ ] Off-target dashboard, ablation table
  - [ ] Replication and generalization section
- [ ] Publish machine-readable CSVs and exact plan.yaml, run.json, pack.lock.json

#### **13. Data and Code Release**
**Status**: ❌ **MISSING**  
**Priority**: 🚨 **CRITICAL**

- [ ] Ship minimal reproducible bundle:
  - [ ] `data/sample_items.jsonl` (small, licensable subset)
  - [ ] Two ready packs
  - [ ] Makefile target `make sample-report` that regenerates PDF locally

#### **14. Safety and Ethics**
**Status**: ❌ **MISSING**  
**Priority**: 🚨 **CRITICAL**

- [ ] Tag packs with risk levels. Only allow low-risk packs in demo mode
- [ ] Add prominent "research only" flag that must be set to run higher-risk objective effects

#### **15. QA Tests that Enforce Rigor**
**Status**: ❌ **MISSING**  
**Priority**: 🚨 **CRITICAL**

- [ ] Unit test that Latin square and blinding are actually applied in test runner
- [ ] Schema test that all packs validate and hash deterministically
- [ ] Golden-master test that analysis pipeline reproduces same CSVs/figures on sample bundle
- [ ] Backend test that ActivationEffect is rejected on API backends with clear error

---

## 🚨 **HIGH PRIORITY (Essential for Paper)**
**Status**: ❌ **MISSING**  
**Paper Requirement**: "Primary: Llama‑3.1‑70B, Qwen‑2.5‑Omni‑7B, Mixtral‑8×22B (MoE)"

#### **Tasks:**
- [ ] Add support for Llama-3.1-70B model
- [ ] Add support for Qwen-2.5-Omni-7B model  
- [ ] Add support for Mixtral-8×22B (MoE) model
- [ ] Implement vLLM integration for throughput optimization
- [ ] Add proper model loading and configuration management
- [ ] Implement model-specific attention hook paths
- [ ] Add device mapping and memory optimization

#### **Files to Modify:**
- `neuromod/neuromod_tool.py` - Add model loading
- `neuromod/testing/test_runner.py` - Add model selection
- `requirements.txt` - Add vLLM dependency

---

### **2. Secondary Benchmarks (Section 4.5.2-4.5.4)**
**Status**: ❌ **MISSING**  
**Paper Requirement**: Cognitive/task battery, telemetry, safety/factuality audit

#### **Tasks:**
- [ ] **Cognitive Tasks Implementation:**
  - [ ] Math/logic short problems
  - [ ] Instruction adherence testing
  - [ ] Summarization brevity tasks
  - [ ] Creative divergence tasks
  - [ ] Focused reasoning battery

- [ ] **Telemetry System:**
  - [ ] Repetition rate calculation
  - [ ] Perplexity slope analysis
  - [ ] Length/entropy metrics
  - [ ] Attention entropy (if available)
  - [ ] KV occupancy tracking

- [ ] **Safety/Factuality Audit:**
  - [ ] Refusal rate measurement
  - [ ] Policy adherence testing
  - [ ] QA factuality sampling
  - [ ] Safety threshold preservation

#### **Files to Create:**
- `neuromod/testing/cognitive_tasks.py`
- `neuromod/testing/telemetry.py`
- `neuromod/testing/safety_audit.py`

---

### **3. Experimental Design Implementation (Section 4.4)**
**Status**: ❌ **MISSING**  
**Paper Requirement**: "Double‑blind, placebo‑controlled, randomized within‑model crossover"

#### **Tasks:**
- [ ] Implement Latin square randomization
- [ ] Add proper crossover design management
- [ ] Implement seed management for replication
- [ ] Add standardized token windows for timing
- [ ] Create condition assignment system
- [ ] Add replication tracking

#### **Files to Modify:**
- `neuromod/testing/test_runner.py` - Add experimental design
- `neuromod/testing/experimental_design.py` - New file for design logic

---

### **4. Human Reference Data Integration (Section 4.6)**
**Status**: ❌ **MISSING**  
**Paper Requirement**: "Match to human signature: cosine/canonical correlation between model subscale vectors and human placebo‑controlled deltas"

#### **Tasks:**
- [ ] Source human psychometric reference data
- [ ] Implement signature matching algorithms
- [ ] Add canonical correlation analysis
- [ ] Create human-model comparison framework
- [ ] Add reference data validation

#### **Files to Create:**
- `neuromod/testing/human_reference.py`
- `neuromod/testing/signature_matching.py`

---

## ⚠️ **MEDIUM PRIORITY (Important for Rigor)**

### **5. Advanced Neuromodulation Effects (Section 4.2)**
**Status**: ⚠️ **PARTIALLY MISSING**  
**Paper Requirement**: KV-cache operations, attention manipulation, steering vectors

#### **Tasks:**
- [ ] **KV-Cache Operations:**
  - [ ] Implement `decay(γ)` function
  - [ ] Implement `stride-compress(s)` function
  - [ ] Implement `truncate(N)` function
  - [ ] Add segment gain functionality

- [ ] **Attention Manipulation:**
  - [ ] Implement head masking with keep_prob
  - [ ] Add optional QK scale proxy
  - [ ] Implement attention sink management

- [ ] **Steering Vector Construction:**
  - [ ] Create contrastive prompt system
  - [ ] Implement layer-wise Δh calculation
  - [ ] Add runtime addition at last-token
  - [ ] Implement storage and retrieval

- [ ] **MoE Router Biases:**
  - [ ] Add router bias modification for Mixtral
  - [ ] Implement expert selection steering

#### **Files to Modify:**
- `neuromod/effects.py` - Add new effects
- `neuromod/neuromod_tool.py` - Add effect application

---

### **6. Advanced Statistical Features (Section 4.7)**
**Status**: ⚠️ **PARTIALLY MISSING**  
**Current**: Basic statistics implemented

#### **Tasks:**
- [ ] **Mixed-Effects Models:**
  - [ ] Full implementation of mixed-effects models
  - [ ] Random intercepts for prompt/set and seed
  - [ ] Fixed effect = condition
  - [ ] Proper model specification and fitting

- [ ] **Bayesian Hierarchical Models:**
  - [ ] Implement Bayesian model framework
  - [ ] Add credible intervals
  - [ ] Implement model comparison (BIC/AIC)

- [ ] **Canonical Correlation:**
  - [ ] Add canonical correlation analysis
  - [ ] Implement human-model signature matching
  - [ ] Add correlation significance testing

#### **Files to Modify:**
- `neuromod/testing/statistical_analysis.py` - Add advanced models

---

### **7. Results Templates & Visualization (Section 5)**
**Status**: ❌ **MISSING**  
**Paper Requirement**: Specific figures and tables

#### **Tasks:**
- [ ] **Figure 1**: Schematic of neuromodulation pack pipeline
- [ ] **Figure 2**: ROC curves for PDQ‑S/SDQ vs placebo per model
- [ ] **Figure 3**: Radar plots of subscale signatures (model vs human)
- [ ] **Figure 4**: Task delta bars (focus/creativity/latency)
- [ ] **Table 1**: Mixed‑effects estimates with 95% CIs

#### **Files to Create:**
- `neuromod/testing/visualization.py` - All plotting functions
- `neuromod/testing/results_templates.py` - Results formatting

---

## 🔧 **LOW PRIORITY (Polish & Documentation)**

### **8. Implementation & Reproducibility (Section 4.8)**
**Status**: ⚠️ **PARTIALLY COMPLETE**

#### **Tasks:**
- [ ] Add environment lockfiles (requirements.txt, environment.yml)
- [ ] Implement deterministic generation where feasible
- [ ] Create BibTeX reading pack
- [ ] Add comprehensive documentation
- [ ] Add reproducibility scripts

#### **Files to Create:**
- `environment.yml` - Conda environment
- `reproducibility.md` - Reproducibility guide
- `BIBLIOGRAPHY.bib` - BibTeX references

---

### **9. Code Quality & Testing**
**Status**: ⚠️ **NEEDS IMPROVEMENT**

#### **Tasks:**
- [ ] Add comprehensive unit tests for new features
- [ ] Add integration tests for experimental design
- [ ] Add performance benchmarks
- [ ] Improve error handling and logging
- [ ] Add type hints throughout codebase

---

## 📁 **File Structure for New Components**

### **Scientific Rigor Foundation (Phase 0)**
```
analysis/
├── plan.yaml                  # Preregistered study plan
├── power_analysis.py          # Power calculation script
└── rigor_checklist.py         # MVR validation

neuromod/testing/
├── rigor/                     # Scientific rigor components
│   ├── __init__.py
│   ├── preregistration.py     # Study planning and validation
│   ├── provenance.py          # Locks, hashes, and run tracking
│   ├── randomization.py       # Latin square and blinding
│   ├── effect_boundaries.py   # Effect type enforcement
│   ├── baselines.py           # Control condition management
│   ├── power_analysis.py      # Sample size calculations
│   ├── off_target.py          # Safety and drift monitoring
│   ├── robustness.py          # Generalization testing
│   ├── ablations.py           # Effect ablation analysis
│   ├── reproducibility.py     # Seed management and caching
│   └── reporting.py           # PDF generation and exports
├── cognitive_tasks.py          # Cognitive task battery
├── telemetry.py               # Performance telemetry
├── safety_audit.py            # Safety and factuality testing
├── human_reference.py         # Human reference data
├── signature_matching.py      # Signature matching algorithms
├── experimental_design.py     # Experimental design logic
├── visualization.py           # All plotting functions
└── results_templates.py       # Results formatting

runs/                          # Run tracking and data
├── <run_id>/
│   ├── run.json               # Run ledger
│   ├── counterbalance.json    # Latin square order
│   ├── key/
│   │   └── unblind.json      # Blinding key
│   ├── prompts/               # Cached prompts
│   ├── outputs/               # Cached outputs
│   └── freeze.txt             # Dependency snapshot

packs/
├── none.json                  # Control condition pack
├── placebo.json               # Placebo pack
└── pack.lock.json             # Pack hashes and versions
```

---

## 🎯 **Implementation Strategy**

### **Phase 0: Scientific Rigor Foundation (Weeks 1-2) - CRITICAL**
1. **Week 1**: Preregistration, locks/provenance, randomization/blinding
   - Create `analysis/plan.yaml` with all MVR requirements
   - Implement `pack.lock.json` and `run.json` ledger system
   - Implement Latin square randomization and blinding system
   - Add automatic leakage detection

2. **Week 2**: Backends, baselines, power analysis
   - Enforce effect type boundaries and application order
   - Implement three-condition baseline system (control, persona, pack)
   - Create power analysis script with pilot study support
   - Implement off-target monitoring system

### **Phase 1: Core Functionality (Weeks 3-4)**
1. Model support (Llama, Qwen, Mixtral)
2. Basic cognitive tasks
3. Telemetry system

### **Phase 2: Experimental Design (Weeks 5-6)**
1. Latin square randomization (already implemented in Phase 0)
2. Crossover design
3. Replication management

### **Phase 3: Advanced Features (Weeks 7-8)**
1. Advanced neuromodulation effects
2. Human reference data integration
3. Advanced statistical models

### **Phase 4: Visualization & Polish (Weeks 9-10)**
1. All figures and tables
2. Documentation
3. Reproducibility scripts

### **Phase 5: Rigor Validation (Week 11)**
1. Run all 15 MVR checklist items
2. Generate sample report with `make sample-report`
3. Validate reproducibility with golden-master tests
4. Final QA testing for all rigor requirements

---

## 🔍 **Validation Checklist**

### **Scientific Rigor Validation (MUST PASS BEFORE PAPER SUBMISSION)**
- [ ] **MVR Checklist Complete**: All 15 points implemented and tested
- [ ] **Preregistration**: `analysis/plan.yaml` created and committed before any experiments
- [ ] **Provenance**: `pack.lock.json` and `run.json` ledger system working
- [ ] **Randomization**: Latin square and blinding properly implemented
- [ ] **Effect Boundaries**: API backends reject ActivationEffects with clear errors
- [ ] **Baselines**: Three-condition system (control, persona, pack) working
- [ ] **Power Analysis**: Pilot studies and n_min calculations working
- [ ] **Off-target Monitoring**: Safety bands enforced and reported
- [ ] **Reproducibility**: `set_run_seed()` and deterministic composition working
- [ ] **Reporting**: PDF generation and machine-readable exports working

### **Core Functionality Validation**
- [ ] All 8 psychometric tests working with new models
- [ ] Cognitive task battery implemented and validated
- [ ] Telemetry system providing meaningful metrics
- [ ] Experimental design properly randomized
- [ ] Statistical analysis includes all required models
- [ ] All figures and tables generated
- [ ] Human reference data integrated
- [ ] Reproducibility scripts working
- [ ] Documentation complete

### **QA Tests for Rigor Enforcement**
- [ ] Latin square and blinding actually applied in test runner
- [ ] All packs validate and hash deterministically
- [ ] Analysis pipeline reproduces same results on sample bundle
- [ ] ActivationEffect rejected on API backends with clear error
- [ ] Sample report regenerates correctly with `make sample-report`

---

## 📚 **References from Paper**

- **Section 4.1**: Llama-3.1-70B, Qwen-2.5-Omni-7B, Mixtral-8×22B
- **Section 4.2**: KV-cache operations, attention manipulation, steering vectors
- **Section 4.4**: Latin square, crossover, replication
- **Section 4.5**: Cognitive tasks, telemetry, safety audit
- **Section 4.6**: Human signature matching
- **Section 4.7**: Mixed-effects, Bayesian, canonical correlation
- **Section 5**: All figures and tables
- **Section 4.8**: Reproducibility and documentation

---

*Last Updated: [Current Date]*
*Status: Active Development*
