# Qwen3-14B Interpretability Study: Quick Start Checklist

**Goal**: Find neural correlates for Solve, Recognize, Generate abilities + concept circuits in Qwen3-14B

---

## Phase 1: Setup (Week 1 - Days 1-2)

### Day 1: Environment Setup

- [ ] **1.1 Check System Requirements**
  ```bash
  # Check Python version (need 3.9+)
  python --version
  
  # Check GPU availability
  nvidia-smi
  
  # Check available disk space (need 100GB+)
  df -h
  ```

- [ ] **1.2 Install/Update Dependencies**
  ```bash
  cd /home/hokindeng/KnowThat-Neuro
  
  # Update transformers for Qwen3 support
  pip install --upgrade transformers>=4.51.0
  
  # Test Qwen3 compatibility
  python -c "from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-14B'); print('✓ Qwen3 supported!')"
  
  # Install analysis dependencies
  pip install torch numpy pandas matplotlib seaborn scikit-learn scipy tqdm
  ```

- [ ] **1.3 Create Directory Structure**
  ```bash
  cd external/three-mountain-interpretability
  
  # Create Qwen3 branch
  git checkout -b qwen3-adaptation
  
  # Create directories
  mkdir -p utils/qwen3/
  mkdir -p data/maze_tasks/
  mkdir -p plots/qwen3_analysis/{dape,ablation,linear_probes,advanced}
  ```

### Day 2: Data Preparation

- [ ] **2.1 Identify Your Existing Data**
  ```bash
  # List your experimental data
  ls -lh results/
  ls -lh data/
  
  # Document what you have:
  # - Recognition tasks: [path]
  # - Generation tasks: [path]
  # - Solving tasks: [path]
  ```

- [ ] **2.2 Create Data Adapter**
  ```bash
  # Copy template
  cp docs/QWEN3_INTERPRETABILITY_PLAN.md utils/qwen3/README.md
  
  # Create adapter file
  touch utils/qwen3/knowthat_data_adapter.py
  # TODO: Implement KnowThatDataAdapter class (see Section 3.2 of plan)
  ```

- [ ] **2.3 Create Task Expander**
  ```bash
  touch utils/qwen3/maze_task_expander.py
  # TODO: Implement MazeTaskExpander class (see Section 3.3 of plan)
  ```

- [ ] **2.4 Test Data Loading**
  ```bash
  # Create test script
  cat > test_data_loading.py << 'EOF'
from utils.qwen3.knowthat_data_adapter import KnowThatDataAdapter

adapter = KnowThatDataAdapter(data_root="data/maze_tasks")
recognize = adapter.load_recognition_tasks()
generate = adapter.load_generation_tasks()
solve = adapter.load_solving_tasks()

print(f"✓ Recognize: {len(recognize)} samples")
print(f"✓ Generate: {len(generate)} samples")
print(f"✓ Solve: {len(solve)} samples")
EOF
  
  python test_data_loading.py
  ```

---

## Phase 2: DAPE Analysis (Week 2-3)

### Week 2: Implementation

- [ ] **3.1 Create DAPE Analyzer**
  ```bash
  touch utils/qwen3/qwen3_dape_analyzer.py
  # TODO: Implement Qwen3DAPEAnalyzer (see Section 4.1 of plan)
  ```

- [ ] **3.2 Create Runner Script**
  ```bash
  touch run_qwen3_dape_analysis.py
  # TODO: Copy template from plan Section 5.3.2
  chmod +x run_qwen3_dape_analysis.py
  ```

- [ ] **3.3 Quick Test (10 samples)**
  ```bash
  python run_qwen3_dape_analysis.py --max-samples 10
  # Expected: ~5-10 minutes, basic outputs
  ```

### Week 3: Full Analysis

- [ ] **3.4 Run Full DAPE Analysis**
  ```bash
  # Full run (200 samples per task)
  python run_qwen3_dape_analysis.py --max-samples 200
  # Expected: 2-4 hours
  ```

- [ ] **3.5 Review DAPE Results**
  ```bash
  ls plots/qwen3_analysis/dape/
  
  # Check key files:
  # - dape_distribution_by_ability.png
  # - layer_distribution_heatmap.png
  # - thinking_vs_direct_comparison.png
  # - qwen3_dape_results.json
  ```

- [ ] **3.6 Extract Key Layers**
  ```python
  import json
  with open('plots/qwen3_analysis/dape/qwen3_dape_results.json') as f:
      results = json.load(f)
  
  # Identify critical layers for next phase
  key_layers = {
      'recognize': results['recognize']['top_layers'][:5],
      'generate': results['generate']['top_layers'][:5],
      'solve': results['solve']['top_layers'][:5],
  }
  print(key_layers)
  # Save these for ablation phase
  ```

---

## Phase 3: Layer Ablation (Week 3)

- [ ] **4.1 Create Layer Ablator**
  ```bash
  touch utils/qwen3/qwen3_layer_ablator.py
  # TODO: Implement Qwen3LayerAblator (see Section 4.2 of plan)
  ```

- [ ] **4.2 Create Runner Script**
  ```bash
  touch run_qwen3_layer_ablation.py
  chmod +x run_qwen3_layer_ablation.py
  ```

- [ ] **4.3 Quick Test**
  ```bash
  python run_qwen3_layer_ablation.py --max-samples 20
  ```

- [ ] **4.4 Run Full Ablation**
  ```bash
  # Use key layers from DAPE
  python run_qwen3_layer_ablation.py --max-samples 100 --key-layers "30,33,36,38,39"
  # Expected: 1-2 hours
  ```

- [ ] **4.5 Review Ablation Results**
  ```bash
  ls plots/qwen3_analysis/ablation/
  
  # Check:
  # - performance_drop_comparison.png
  # - layer_sensitivity_heatmap.png
  # - ablation_results.json
  ```

---

## Phase 4: Linear Probes (Week 4)

- [ ] **5.1 Create Probe Analyzer**
  ```bash
  touch utils/qwen3/qwen3_linear_probe_analyzer.py
  # TODO: Implement Qwen3LinearProbeAnalyzer (see Section 4.3 of plan)
  ```

- [ ] **5.2 Generate Balanced Dataset**
  ```bash
  # Create and run expander
  python -c "
from utils.qwen3.maze_task_expander import MazeTaskExpander
from utils.qwen3.knowthat_data_adapter import KnowThatDataAdapter

adapter = KnowThatDataAdapter(data_root='data/maze_tasks')
samples = adapter.load_all_tasks()

expander = MazeTaskExpander()
balanced = expander.expand_all_tasks(samples, max_samples_per_task=500)

print(f'Total balanced samples: {len(balanced)}')
# Save for probe training
"
  ```

- [ ] **5.3 Create Runner Script**
  ```bash
  touch run_qwen3_linear_probes.py
  chmod +x run_qwen3_linear_probes.py
  ```

- [ ] **5.4 Quick Test**
  ```bash
  python run_qwen3_linear_probes.py --max-samples 50
  ```

- [ ] **5.5 Run Full Probe Analysis**
  ```bash
  python run_qwen3_linear_probes.py --max-samples 500 --use-thinking-states
  # Expected: 2-3 hours
  ```

- [ ] **5.6 Review Probe Results**
  ```bash
  ls plots/qwen3_analysis/linear_probes/
  
  # Check key visualizations:
  # - layer_performance_curves.png
  # - confidence_analysis.png
  # - information_flow_analysis.png
  # - probe_results.json
  ```

---

## Phase 5: Advanced Analysis (Week 5)

- [ ] **6.1 Thinking Trace Analysis**
  ```bash
  touch utils/qwen3/thinking_trace_analyzer.py
  # TODO: Implement ThinkingTraceAnalyzer (see Section 4.4 of plan)
  ```

- [ ] **6.2 Concept Circuit Analysis**
  ```bash
  touch utils/qwen3/concept_circuit_analyzer.py
  # TODO: Implement ConceptCircuitAnalyzer (see Section 4.5 of plan)
  ```

- [ ] **6.3 Label Samples with Concepts**
  ```bash
  # Extend your data adapter to include concept labels
  # Concepts: wall, path, turn, dead_end, solution, start, goal
  
  # Add to KnowThatDataAdapter:
  # def load_all_tasks_with_concepts(self):
  #     ...
  ```

- [ ] **6.4 Run Advanced Analysis**
  ```bash
  touch run_qwen3_advanced_analysis.py
  chmod +x run_qwen3_advanced_analysis.py
  
  python run_qwen3_advanced_analysis.py
  # Expected: 1-2 hours
  ```

- [ ] **6.5 Review Advanced Results**
  ```bash
  ls plots/qwen3_analysis/advanced/
  
  # Check:
  # - reasoning_neurons.png
  # - concept_circuits.png
  # - circuit_connectivity.png
  ```

---

## Phase 6: Synthesis (Week 5-6)

- [ ] **7.1 Load All Results**
  ```bash
  touch synthesize_qwen3_results.py
  chmod +x synthesize_qwen3_results.py
  ```

- [ ] **7.2 Run Synthesis**
  ```bash
  python synthesize_qwen3_results.py
  ```

- [ ] **7.3 Review Synthesis Report**
  ```bash
  cat plots/qwen3_analysis/SYNTHESIS_REPORT.txt
  
  # Should show:
  # - Convergent critical layers (all methods agree)
  # - Hierarchical specialization patterns
  # - Thinking mode impact
  # - Key findings for paper
  ```

- [ ] **7.4 Create Publication Figures**
  ```bash
  # Create final high-quality figures
  python create_publication_figures.py
  
  # Output: plots/qwen3_analysis/paper_figures/
  ```

---

## Verification Checkpoints

### Checkpoint 1: Data Ready (End of Week 1)
- [ ] Have 2,400+ samples (3 tasks × 800 samples)
- [ ] Each sample has correct ability labels
- [ ] Data loading tests pass
- [ ] Balanced positive/negative samples

### Checkpoint 2: DAPE Complete (End of Week 3)
- [ ] Identified 5-15% domain-specific neurons
- [ ] Clear layer distribution patterns
- [ ] Thinking vs direct mode comparison done
- [ ] Key layers extracted for ablation

### Checkpoint 3: Ablation Complete (End of Week 3)
- [ ] Critical layers identified (5-8 per ability)
- [ ] Performance drops measured
- [ ] Hierarchical dependency validated

### Checkpoint 4: Probes Complete (End of Week 4)
- [ ] Probe accuracy 65-85% (not trivial)
- [ ] Performance curves across layers
- [ ] Cross-task generalization tested
- [ ] All 8 enhanced analyses done

### Checkpoint 5: Advanced Complete (End of Week 5)
- [ ] Reasoning-specific neurons identified
- [ ] Concept circuits mapped
- [ ] Circuit connectivity visualized

### Checkpoint 6: Synthesis Complete (End of Week 6)
- [ ] All methods converge on critical layers
- [ ] Synthesis report written
- [ ] Publication figures ready
- [ ] Key insights documented

---

## Quick Reference: File Checklist

### Files to Create

```
utils/qwen3/
├── __init__.py
├── knowthat_data_adapter.py          [ ] Created [ ] Tested
├── maze_task_expander.py              [ ] Created [ ] Tested
├── qwen3_dape_analyzer.py             [ ] Created [ ] Tested
├── qwen3_layer_ablator.py             [ ] Created [ ] Tested
├── qwen3_linear_probe_analyzer.py     [ ] Created [ ] Tested
├── thinking_trace_analyzer.py         [ ] Created [ ] Tested
└── concept_circuit_analyzer.py        [ ] Created [ ] Tested

Root scripts:
├── test_data_loading.py               [ ] Created [ ] Tested
├── run_qwen3_dape_analysis.py         [ ] Created [ ] Tested
├── run_qwen3_layer_ablation.py        [ ] Created [ ] Tested
├── run_qwen3_linear_probes.py         [ ] Created [ ] Tested
├── run_qwen3_advanced_analysis.py     [ ] Created [ ] Tested
└── synthesize_qwen3_results.py        [ ] Created [ ] Tested

Config:
├── config/qwen3_config.py             [ ] Created
└── config/concept_labels.json         [ ] Created
```

---

## Troubleshooting

### Issue: CUDA Out of Memory
```bash
# Solution 1: Reduce batch size
python run_qwen3_dape_analysis.py --max-samples 50

# Solution 2: Use gradient checkpointing
# Add to model loading: model.gradient_checkpointing_enable()

# Solution 3: Use CPU offloading
# device_map="auto" when loading model
```

### Issue: Probe Accuracy Too Low (<60%)
```bash
# Check label balance
python -c "
import pandas as pd
from utils.qwen3.knowthat_data_adapter import KnowThatDataAdapter
adapter = KnowThatDataAdapter(data_root='data/maze_tasks')
samples = adapter.load_all_tasks()
df = pd.DataFrame(samples)
print(df[['requires_recognize', 'requires_generate', 'requires_solve']].value_counts())
"

# If imbalanced, regenerate balanced data
```

### Issue: Probe Accuracy Too High (>95%)
```bash
# You have the task-ability correlation artifact
# Need to generate negative samples

# Check if expander was used
# Should have both 0s and 1s for each ability within each task type
```

---

## Daily Time Estimates

| Day | Phase | Tasks | Time |
|-----|-------|-------|------|
| 1-2 | Setup + Data | Environment, adapter, expander | 4-6 hours |
| 3-7 | DAPE | Implementation + testing | 2-3 hours/day |
| 8-10 | DAPE Full | Running full analysis | 4-6 hours (mostly compute) |
| 11-13 | Ablation | Implementation + full run | 3-4 hours/day |
| 14-20 | Probes | Implementation + all analyses | 3-4 hours/day |
| 21-25 | Advanced | Thinking + circuits | 3-4 hours/day |
| 26-30 | Synthesis | Cross-method validation + report | 3-4 hours/day |

**Total Human Time**: ~80-100 hours  
**Total Compute Time**: ~40-60 GPU-hours  
**Total Calendar Time**: 6 weeks (part-time) or 3 weeks (full-time)

---

## Success Metrics

At the end, you should have:

✅ **Quantitative Results**:
- [ ] DAPE scores for all neurons across 40 layers
- [ ] Critical layers identified for each ability
- [ ] Probe accuracies: 65-85%
- [ ] Performance drops: 15-50% when ablating key layers
- [ ] 3-5 layers where all methods agree per ability

✅ **Qualitative Insights**:
- [ ] Can explain which layers handle which abilities
- [ ] Can map concept circuits for maze elements
- [ ] Can identify reasoning-specific neurons
- [ ] Can predict performance from layer ablation

✅ **Deliverables**:
- [ ] 20-30 publication-quality figures
- [ ] Synthesis report with key findings
- [ ] JSON files with all numerical results
- [ ] Reproducible code + documentation

---

## Next Actions (Right Now!)

**Start here:**

1. [ ] Open terminal and run checkpoint 1.1 (check system)
2. [ ] Install/update dependencies (checkpoint 1.2)
3. [ ] Create directory structure (checkpoint 1.3)
4. [ ] Identify your existing data (checkpoint 2.1)
5. [ ] Read Section 3 of QWEN3_INTERPRETABILITY_PLAN.md carefully
6. [ ] Start implementing `knowthat_data_adapter.py`

**First Milestone Target**: End of Week 1
- Working data pipeline
- 2,400+ balanced samples ready
- Data loading tests pass

**Go! 🚀**

---

**Last Updated**: November 2024  
**Estimated Completion**: 6 weeks from start  
**Required GPU**: 1× A100 (40-80GB) or 2× A6000  
**Total Compute**: ~40-60 GPU-hours

