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

## 🚨 **HIGH PRIORITY (Essential for Paper)**

### **1. Models and Serving Stacks (Section 4.1)**
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

```
neuromod/testing/
├── cognitive_tasks.py          # Cognitive task battery
├── telemetry.py               # Performance telemetry
├── safety_audit.py            # Safety and factuality testing
├── human_reference.py         # Human reference data
├── signature_matching.py      # Signature matching algorithms
├── experimental_design.py     # Experimental design logic
├── visualization.py           # All plotting functions
└── results_templates.py       # Results formatting
```

---

## 🎯 **Implementation Strategy**

### **Phase 1: Core Functionality (Weeks 1-2)**
1. Model support (Llama, Qwen, Mixtral)
2. Basic cognitive tasks
3. Telemetry system

### **Phase 2: Experimental Design (Weeks 3-4)**
1. Latin square randomization
2. Crossover design
3. Replication management

### **Phase 3: Advanced Features (Weeks 5-6)**
1. Advanced neuromodulation effects
2. Human reference data integration
3. Advanced statistical models

### **Phase 4: Visualization & Polish (Weeks 7-8)**
1. All figures and tables
2. Documentation
3. Reproducibility scripts

---

## 🔍 **Validation Checklist**

Before considering the paper implementation complete, verify:

- [ ] All 8 psychometric tests working with new models
- [ ] Cognitive task battery implemented and validated
- [ ] Telemetry system providing meaningful metrics
- [ ] Experimental design properly randomized
- [ ] Statistical analysis includes all required models
- [ ] All figures and tables generated
- [ ] Human reference data integrated
- [ ] Reproducibility scripts working
- [ ] Documentation complete

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
