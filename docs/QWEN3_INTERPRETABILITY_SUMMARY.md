# Qwen3-14B Interpretability Study: Executive Summary

**Target**: Identify neural correlates for **Solve**, **Recognize**, **Generate** + concept circuits in Qwen3-14B

---

## What You're Doing

Adapting the **Three Mountain Interpretability** framework (originally for Qwen2.5-VL) to study **Qwen3-14B** on your maze navigation experiments.

### Three Research Questions

1. **Which neurons specialize for each cognitive ability?**
   - Method: DAPE (Domain Activation Probability Entropy)
   - Answer: "Neuron #2345 in layer 35 activates 90% of the time for 'solve' tasks"

2. **Which layers are critical for each ability?**
   - Method: Layer Ablation
   - Answer: "Layers 32-40 are critical for solving (40% performance drop when ablated)"

3. **How is cognitive information encoded across layers?**
   - Method: Linear Probes
   - Answer: "Recognize information peaks at layer 20, Solve peaks at layer 38"

---

## Your Cognitive Ability Hierarchy

```
Level 3: SOLVE (Layers 32-40)
    ↓ requires
Level 2: GENERATE (Layers 22-35) + Level 1
    ↓ requires  
Level 1: RECOGNIZE (Layers 12-24)
    ↓ requires
Level 0: Basic Pattern Recognition
```

### Task Examples

**Recognize**: "Is this path valid?" "Does this maze have a solution?"  
**Generate**: "Create a 5×5 maze" "Draw an alternative path"  
**Solve**: "Find the shortest path step-by-step"

---

## Three Analysis Methods

### 1. DAPE Analysis (Neuron-Level)
**What**: Identifies neurons that specialize for specific abilities  
**How**: Measures activation entropy across tasks (low entropy = specialized)  
**Output**: 
- List of specialized neurons per ability
- Layer distribution heatmaps
- Thinking vs direct mode comparison

**Expected Finding**: 5-15% of neurons are domain-specific, concentrated in late layers

### 2. Layer Ablation (Layer-Level)
**What**: Identifies critical layers through causal intervention  
**How**: Zero out layer activations and measure performance drop  
**Output**:
- Critical layer identification
- Performance degradation curves
- Hierarchical dependency validation

**Expected Finding**: Different abilities rely on different layer ranges

### 3. Linear Probes (Representation-Level)
**What**: Detects which layers encode ability information  
**How**: Train binary classifiers on layer activations  
**Output**:
- Performance curves across layers
- Confidence evolution
- Information flow analysis
- Cross-task generalization

**Expected Finding**: Information accumulates gradually, peaks at different layers

---

## Qwen3-14B Special Features

### 1. Thinking Mode
Qwen3 can generate explicit reasoning traces: `<think>reasoning</think>answer`

**New Analysis**: Compare neural activations in thinking vs direct mode
- Reasoning-specific neurons (active only in thinking region)
- Broader layer engagement in thinking mode
- Different neuron populations recruited

### 2. Text-Only (No Vision)
Simpler than Qwen2.5-VL, easier to analyze
- Remove vision preprocessing
- Focus on language processing
- Cleaner attribution signals

### 3. 40 Layers (vs 36)
More granular layer-wise analysis
- Better layer resolution
- More specialized late layers
- Extended hierarchy

---

## Implementation Plan

### Phase 1: Data Preparation (Week 1)
**Tasks**:
- Extract your experimental data (recognition, generation, solving tasks)
- Create data adapter: `KnowThatDataAdapter`
- Generate balanced negatives: `MazeTaskExpander`
- Test data loading

**Deliverable**: 2,400+ balanced samples ready

### Phase 2: DAPE Analysis (Week 2-3)
**Tasks**:
- Implement `Qwen3DAPEAnalyzer`
- Collect neuron activations across all layers
- Calculate DAPE scores
- Identify domain-specific neurons
- Compare thinking vs direct mode

**Deliverable**: Domain-specific neuron lists + visualizations

### Phase 3: Layer Ablation (Week 3)
**Tasks**:
- Implement `Qwen3LayerAblator`
- Systematically ablate layers at different ratios
- Measure performance degradation
- Identify critical layers per ability

**Deliverable**: Critical layer identification + validation

### Phase 4: Linear Probes (Week 4)
**Tasks**:
- Implement `Qwen3LinearProbeAnalyzer`
- Extract representations from all 40 layers
- Train probes for each ability
- Run 8 enhanced analyses (confidence, PCA, errors, etc.)
- Test cross-task generalization

**Deliverable**: Probe results + information flow analysis

### Phase 5: Advanced Analysis (Week 5)
**Tasks**:
- Thinking trace analysis (reasoning-specific neurons)
- Concept circuit identification (wall, path, turn, etc.)
- Circuit connectivity visualization

**Deliverable**: Concept circuit maps + reasoning neuron analysis

### Phase 6: Synthesis (Week 5-6)
**Tasks**:
- Cross-method validation
- Find convergent critical layers
- Write synthesis report
- Create publication figures

**Deliverable**: Final interpretability report + paper figures

---

## Expected Results

### Hierarchical Specialization
```
Recognize: Layers 12-24 (pattern detection)
    - DAPE: 8-12% specialized neurons
    - Ablation: 20-30% performance drop
    - Probes: 75-82% accuracy peak

Generate: Layers 22-35 (structural synthesis)
    - DAPE: 10-15% specialized neurons
    - Ablation: 30-40% performance drop
    - Probes: 70-80% accuracy peak

Solve: Layers 32-40 (sequential reasoning)
    - DAPE: 12-18% specialized neurons
    - Ablation: 40-50% performance drop
    - Probes: 65-75% accuracy peak
```

### Thinking Mode Impact
```
Thinking Mode:
- Broader layer range (15-40)
- More specialized neurons activated
- Higher confidence in late layers
- Explicit reasoning in layers 30-38

Direct Mode:
- Narrower layer range (28-40)
- More general neurons
- Lower confidence
- Compressed reasoning
```

### Concept Circuits
```
"Wall" neurons: Layers 8-25 (boundary detection)
"Path" neurons: Layers 20-35 (trajectory encoding)
"Solution" neurons: Layers 35-40 (goal satisfaction)
```

### Convergence
All three methods will agree on 70%+ of critical layers, giving high confidence in findings.

---

## Resource Requirements

### Computational
- **GPU**: 1× A100 (40GB) minimum, 1× A100 (80GB) recommended
- **RAM**: 64GB minimum, 128GB recommended
- **Storage**: 100GB (data + results)
- **Time**: 40-60 GPU-hours

### Human
- **Total**: 80-120 hours over 6 weeks
- **Week 1**: 6 hours (setup + data)
- **Week 2-3**: 20 hours (DAPE + ablation)
- **Week 4**: 20 hours (probes)
- **Week 5**: 16 hours (advanced)
- **Week 6**: 16 hours (synthesis)

---

## Success Criteria

### You'll Know You Succeeded When:

✅ **Quantitative**:
- Identified 5-15% specialized neurons per ability
- Found 3-5 critical layers per ability (all methods agree)
- Probe accuracy 65-85% (not trivial)
- Performance drops 15-50% when ablating key layers

✅ **Qualitative**:
- Can explain which neurons/layers handle which abilities
- Can predict performance from layer patterns
- Can map concept circuits for maze elements
- Can identify reasoning-specific neurons

✅ **Deliverables**:
- 20-30 publication-quality figures
- Synthesis report with key findings
- Reproducible code + documentation
- Paper-ready results

---

## Key Innovations Over Original Three Mountain

1. **Thinking Mode Analysis**: Unique to Qwen3, analyze explicit reasoning
2. **Concept Circuits**: Go beyond abilities to map maze concepts
3. **Hierarchical Validation**: Validate Solve depends on Recognize + Generate
4. **Cross-Mode Comparison**: Thinking vs Direct processing
5. **40-Layer Resolution**: Finer-grained layer analysis

---

## What You'll Discover

### Scientific Contributions

1. **Mechanistic Understanding**:
   - First interpretability study of Qwen3-14B
   - Neural correlates of spatial reasoning abilities
   - How thinking mode changes neural processing

2. **Concept Circuits**:
   - Specific neurons for maze concepts (wall, path, turn)
   - Circuit connectivity across layers
   - Validated through ablation

3. **Hierarchical Cognition**:
   - Evidence for compositional reasoning
   - Layer-wise ability specialization
   - Dependency validation (Solve needs Recognize + Generate)

### Practical Applications

1. **Model Improvement**:
   - Target specific layers for fine-tuning
   - Design prompts that activate critical neurons
   - Intervene on concept circuits for better performance

2. **Debugging**:
   - Identify why model fails (which layer/neuron issue)
   - Predict failure modes from activation patterns
   - Guide architecture improvements

3. **Efficiency**:
   - Prune non-critical neurons
   - Focus computation on key layers
   - Optimize inference paths

---

## Timeline Overview

```
Week 1: Setup + Data
├─ Day 1-2: Environment setup
└─ Day 3-5: Data adapter + expander

Week 2-3: DAPE + Ablation
├─ Day 6-10: DAPE implementation + testing
├─ Day 11-13: Full DAPE run + analysis
└─ Day 14-15: Ablation implementation + run

Week 4: Linear Probes
├─ Day 16-18: Probe implementation
├─ Day 19-20: Full probe run
└─ Day 21: Enhanced analyses

Week 5: Advanced
├─ Day 22-23: Thinking trace analysis
├─ Day 24-25: Concept circuits
└─ Day 26: Circuit visualization

Week 6: Synthesis
├─ Day 27-28: Cross-method validation
├─ Day 29: Synthesis report
└─ Day 30: Publication figures
```

---

## Documents Guide

### 📘 THREE_MOUNTAIN_INTERPRETABILITY_GUIDE.md
**Purpose**: Comprehensive overview of the original framework  
**Read When**: Need to understand the original methods  
**Length**: 1,403 lines

### 📗 QWEN3_INTERPRETABILITY_PLAN.md
**Purpose**: Detailed adaptation plan for Qwen3-14B  
**Read When**: Implementing code, understanding methods  
**Length**: 850 lines (this is the main technical document)

### 📕 QWEN3_QUICK_START_CHECKLIST.md
**Purpose**: Day-by-day implementation checklist  
**Read When**: Ready to start coding  
**Length**: Practical checklist format

### 📙 QWEN3_INTERPRETABILITY_SUMMARY.md (this document)
**Purpose**: High-level overview and quick reference  
**Read When**: First time or quick review  
**Length**: Concise summary

---

## Reading Order

### First Time (1 hour)
1. Read this summary (15 min)
2. Skim QWEN3_INTERPRETABILITY_PLAN.md Sections 1-2 (20 min)
3. Review QWEN3_QUICK_START_CHECKLIST.md Phase 1 (15 min)
4. Reference THREE_MOUNTAIN_INTERPRETABILITY_GUIDE.md Section 3 (10 min)

### Before Starting Implementation (2 hours)
1. Read QWEN3_INTERPRETABILITY_PLAN.md Sections 3-5 in detail
2. Study code templates in Section 7
3. Review checklist for your current phase

### During Implementation (ongoing)
1. Use QWEN3_QUICK_START_CHECKLIST.md as daily guide
2. Reference QWEN3_INTERPRETABILITY_PLAN.md for technical details
3. Check THREE_MOUNTAIN_INTERPRETABILITY_GUIDE.md for methodology questions

---

## Quick Reference: Key Code Files to Create

```python
# Week 1: Data
utils/qwen3/knowthat_data_adapter.py      # Load your experimental data
utils/qwen3/maze_task_expander.py         # Generate balanced negatives

# Week 2-3: DAPE + Ablation
utils/qwen3/qwen3_dape_analyzer.py        # Neuron specialization analysis
utils/qwen3/qwen3_layer_ablator.py        # Layer criticality testing

# Week 4: Probes
utils/qwen3/qwen3_linear_probe_analyzer.py # Representation encoding

# Week 5: Advanced
utils/qwen3/thinking_trace_analyzer.py     # Thinking vs direct mode
utils/qwen3/concept_circuit_analyzer.py    # Concept circuit mapping

# Week 6: Synthesis
synthesize_qwen3_results.py                # Cross-method validation
```

---

## Next Steps (Right Now)

1. **Read** QWEN3_QUICK_START_CHECKLIST.md Phase 1
2. **Check** your system requirements (GPU, disk space)
3. **Install** dependencies (`pip install transformers>=4.51.0`)
4. **Identify** your experimental data files
5. **Start** implementing `knowthat_data_adapter.py`

**First Milestone**: End of Week 1 - working data pipeline with 2,400+ samples

---

## Questions to Answer Before Starting

- [ ] Do you have experimental data for recognize, generate, and solve tasks?
- [ ] How many samples do you have per task? (need 200+ per task minimum)
- [ ] What format is your data in? (JSON, Parquet, CSV?)
- [ ] Do you have GPU access? (need A100 or equivalent)
- [ ] Can you dedicate 15-20 hours per week for 6 weeks?

If you answered "yes" to all → **You're ready to start!**

---

## Contact & Support

- **Documentation**: `docs/` folder (4 comprehensive guides)
- **Original Framework**: `external/three-mountain-interpretability/`
- **Example Code**: See Section 7 of QWEN3_INTERPRETABILITY_PLAN.md

**Start Here**: QWEN3_QUICK_START_CHECKLIST.md → Phase 1 → Day 1 → Task 1.1

---

**Good luck! 🚀**

---

**Document**: Executive Summary  
**Last Updated**: November 2024  
**Status**: Ready to Start  
**Next Action**: Open QWEN3_QUICK_START_CHECKLIST.md and begin Phase 1

