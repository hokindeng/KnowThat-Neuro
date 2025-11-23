# Qwen3-14B Interpretability Study Documentation

**Goal**: Find neural correlates for **Solve**, **Recognize**, **Generate** abilities and concept circuits in Qwen3-14B using the Three Mountain Interpretability framework.

---

## 📚 Documentation Overview

I've created **4 comprehensive guides** to help you complete this 6-week interpretability study:

### 1. **QWEN3_INTERPRETABILITY_SUMMARY.md** ← START HERE
**Purpose**: High-level overview and quick reference  
**Read Time**: 15 minutes  
**When to Read**: First time, or when you need a quick refresher  

**Contains**:
- What you're doing (3 research questions)
- Three analysis methods (DAPE, Ablation, Probes)
- Implementation plan overview (6 weeks)
- Expected results summary
- Success criteria

**Start Command**:
```bash
cat docs/QWEN3_INTERPRETABILITY_SUMMARY.md
```

---

### 2. **QWEN3_INTERPRETABILITY_PLAN.md** ← MAIN TECHNICAL GUIDE
**Purpose**: Detailed implementation guide with code templates  
**Read Time**: 2-3 hours  
**When to Read**: Before implementing each phase  

**Contains**:
- Model architecture considerations (Qwen3-14B specs)
- Complete ability definitions (Solve, Recognize, Generate)
- Data preparation strategy
- All three analysis methods with code templates
- 6-week implementation roadmap
- Expected insights and results
- Code adaptation guide from Three Mountain

**Start Command**:
```bash
code docs/QWEN3_INTERPRETABILITY_PLAN.md  # or your editor
```

**Key Sections**:
- Section 3: Data Preparation Strategy → Read first
- Section 4: Analysis Pipeline → Implement week by week
- Section 5: Implementation Roadmap → Your schedule
- Section 7: Code Adaptation Guide → Reference while coding

---

### 3. **QWEN3_QUICK_START_CHECKLIST.md** ← DAILY TODO LIST
**Purpose**: Step-by-step daily tasks with copy-paste commands  
**Read Time**: Reference as needed  
**When to Read**: Every day during implementation  

**Contains**:
- Day-by-day task breakdown
- Terminal commands ready to copy-paste
- Verification checkpoints
- Troubleshooting guide
- Time estimates per task
- Success metrics

**Start Command**:
```bash
cat docs/QWEN3_QUICK_START_CHECKLIST.md | less
```

**How to Use**:
- Check off tasks as you complete them
- Copy-paste terminal commands
- Verify checkpoints at end of each week
- Reference troubleshooting when stuck

---

### 4. **QWEN3_ARCHITECTURE_DIAGRAM.md** ← VISUAL REFERENCE
**Purpose**: Visual diagrams and flowcharts  
**Read Time**: 20 minutes  
**When to Read**: When you need to understand the big picture  

**Contains**:
- Overall pipeline diagram
- Qwen3-14B architecture visualization
- Three analysis methods comparison
- Hierarchical ability dependencies
- Concept circuit examples
- Data flow diagrams
- File organization chart
- Timeline visualization

**Start Command**:
```bash
cat docs/QWEN3_ARCHITECTURE_DIAGRAM.md | less
```

---

## 🎯 Quick Start: What to Read First

### First Time (1 hour)
```bash
# 1. Read summary (15 min)
cat docs/QWEN3_INTERPRETABILITY_SUMMARY.md

# 2. Skim plan sections 1-2 (20 min)
code docs/QWEN3_INTERPRETABILITY_PLAN.md  # Read sections 1-2

# 3. Review checklist Phase 1 (15 min)
cat docs/QWEN3_QUICK_START_CHECKLIST.md | head -200

# 4. View diagrams (10 min)
cat docs/QWEN3_ARCHITECTURE_DIAGRAM.md | less
```

### Before Starting Implementation (2 hours)
```bash
# 1. Read plan sections 3-5 in detail
# Focus on:
#   - Section 3: Data Preparation Strategy
#   - Section 4: Analysis Pipeline
#   - Section 5: Implementation Roadmap

# 2. Study code templates in section 7
# You'll be copying and adapting these

# 3. Print checklist or keep open in terminal
cat docs/QWEN3_QUICK_START_CHECKLIST.md
```

### During Implementation (ongoing)
```bash
# Daily workflow:
# 1. Check checklist for today's tasks
# 2. Reference plan for technical details
# 3. Copy-paste commands from checklist
# 4. Review diagrams when confused
```

---

## 📊 Documentation Comparison Table

| Document | Length | Purpose | Read When | Reference Frequency |
|----------|--------|---------|-----------|---------------------|
| **Summary** | 15 min | Overview | First time | Occasionally |
| **Plan** | 2-3 hours | Technical details | Before each phase | Constantly |
| **Checklist** | Ongoing | Daily tasks | Every day | Daily |
| **Diagrams** | 20 min | Visual reference | When confused | Occasionally |

---

## 🗺️ Recommended Reading Path

### Week 1: Setup + Data
**Before starting**:
1. Read Summary (all)
2. Read Plan Section 3 (Data Preparation)
3. Read Checklist Phase 1 (Week 1)

**During work**:
- Checklist as daily guide
- Plan Section 3 as reference
- Diagrams for data flow

### Week 2-3: DAPE + Ablation
**Before starting**:
1. Read Plan Section 4.1 (DAPE Analysis)
2. Read Plan Section 4.2 (Layer Ablation)
3. Read Checklist Phase 2-3

**During work**:
- Checklist Phase 2-3 as guide
- Plan Section 4.1-4.2 as reference
- Diagrams for method understanding

### Week 4: Linear Probes
**Before starting**:
1. Read Plan Section 4.3 (Linear Probes)
2. Read Checklist Phase 4
3. Review THREE_MOUNTAIN_INTERPRETABILITY_GUIDE.md Section 3.3

**During work**:
- Checklist Phase 4 as guide
- Plan Section 4.3 as reference
- Original guide for probe methodology

### Week 5: Advanced Analysis
**Before starting**:
1. Read Plan Section 4.4 (Thinking Trace)
2. Read Plan Section 4.5 (Concept Circuits)
3. Read Checklist Phase 5

**During work**:
- Checklist Phase 5 as guide
- Plan Section 4.4-4.5 as reference
- Diagrams for circuit visualization

### Week 6: Synthesis
**Before starting**:
1. Read Plan Section 7 (Results & Interpretation)
2. Read Checklist Phase 6

**During work**:
- Checklist Phase 6 as guide
- Summary for expected findings
- Plan Section 7 for interpretation

---

## 🚀 Quick Start Commands

### Step 1: Check Prerequisites (5 minutes)
```bash
# Navigate to project
cd /home/hokindeng/KnowThat-Neuro

# Check Python
python --version  # Need 3.9+

# Check GPU
nvidia-smi  # Need A100 or equivalent

# Check disk space
df -h  # Need 100GB+
```

### Step 2: Read Documentation (1 hour)
```bash
# Read summary
cat docs/QWEN3_INTERPRETABILITY_SUMMARY.md

# Open plan in editor
code docs/QWEN3_INTERPRETABILITY_PLAN.md

# Keep checklist open
cat docs/QWEN3_QUICK_START_CHECKLIST.md | less
```

### Step 3: Setup Environment (2 hours)
```bash
# Update transformers
pip install --upgrade transformers>=4.51.0

# Test Qwen3
python -c "from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-14B'); print('✓ Ready!')"

# Create directories
cd external/three-mountain-interpretability
git checkout -b qwen3-adaptation
mkdir -p utils/qwen3/ data/maze_tasks/ plots/qwen3_analysis/{dape,ablation,linear_probes,advanced}
```

### Step 4: Start Implementation (Week 1)
```bash
# Create data adapter (follow Plan Section 3.2)
touch utils/qwen3/knowthat_data_adapter.py
code utils/qwen3/knowthat_data_adapter.py

# Follow QWEN3_QUICK_START_CHECKLIST.md Phase 1
# You're now on your way!
```

---

## 📖 Section Quick Reference

### From QWEN3_INTERPRETABILITY_PLAN.md

**Section 1**: Model Architecture Considerations  
- Qwen3-14B specs (40 layers, 14.8B params)
- Thinking mode explanation
- Differences from Qwen2.5-VL

**Section 2**: Ability Definitions & Task Mapping  
- Recognize, Generate, Solve definitions
- Hierarchical dependencies
- Task-ability mapping table

**Section 3**: Data Preparation Strategy  
- Data structure format
- KnowThatDataAdapter implementation
- MazeTaskExpander for balanced negatives
- Sample size recommendations

**Section 4**: Analysis Pipeline  
- 4.1: DAPE Analysis (neuron specialization)
- 4.2: Layer Ablation (critical layers)
- 4.3: Linear Probes (information encoding)
- 4.4: Thinking Trace Analysis (NEW)
- 4.5: Concept Circuit Identification (NEW)

**Section 5**: Implementation Roadmap  
- Week-by-week breakdown
- Phase 1-6 detailed tasks
- Milestones and deliverables

**Section 6**: Expected Insights  
- Hierarchical specialization patterns
- Thinking mode impact
- Concept circuits
- Solve dependency validation

**Section 7**: Code Adaptation Guide  
- Key changes from Three Mountain
- Model loading differences
- Input formatting for text-only
- Configuration template

---

## 🎓 Learning Resources

### Background Reading (Optional)
1. **Original Three Mountain Framework**:
   - `docs/THREE_MOUNTAIN_INTERPRETABILITY_GUIDE.md`
   - Focus on Section 3 (Analysis Methods)

2. **Qwen3-14B Model Card**:
   - https://huggingface.co/Qwen/Qwen3-14B
   - Focus on thinking mode usage

3. **Related Work**:
   - Anthropic's Circuits work
   - OpenAI's Microscope
   - Neuron2Graph papers

### When You're Stuck

**Problem**: Don't understand a concept  
**Solution**: Check diagrams in QWEN3_ARCHITECTURE_DIAGRAM.md

**Problem**: Don't know what to do next  
**Solution**: Follow QWEN3_QUICK_START_CHECKLIST.md

**Problem**: Implementation details unclear  
**Solution**: Read relevant section in QWEN3_INTERPRETABILITY_PLAN.md

**Problem**: Code not working  
**Solution**: Check troubleshooting in QWEN3_QUICK_START_CHECKLIST.md Section 9

---

## 📝 Tracking Your Progress

### Checklist Progress Tracker
```bash
# Create a progress file
cat > progress.md << 'EOF'
# Qwen3 Interpretability Progress

## Week 1: Setup + Data
- [ ] Environment setup (Day 1)
- [ ] Data adapter (Day 2-3)
- [ ] Task expander (Day 4)
- [ ] Data loading tests (Day 5)
- [ ] Checkpoint 1: 2,400+ samples ready

## Week 2-3: DAPE + Ablation
- [ ] DAPE analyzer implementation (Week 2)
- [ ] DAPE full run (Week 2)
- [ ] DAPE results analysis (Week 2)
- [ ] Ablation implementation (Week 3)
- [ ] Ablation full run (Week 3)
- [ ] Checkpoint 2: Critical layers identified

## Week 4: Linear Probes
- [ ] Probe analyzer implementation
- [ ] Balanced data generation
- [ ] Probe training (all abilities)
- [ ] Enhanced analyses (8 methods)
- [ ] Checkpoint 3: Probe results complete

## Week 5: Advanced
- [ ] Thinking trace analysis
- [ ] Concept circuit identification
- [ ] Circuit visualization
- [ ] Checkpoint 4: Advanced analyses done

## Week 6: Synthesis
- [ ] Cross-method validation
- [ ] Synthesis report
- [ ] Publication figures
- [ ] Checkpoint 5: Final deliverables ready

## Current Status
**Week**: [1-6]
**Phase**: [Setup/DAPE/Ablation/Probes/Advanced/Synthesis]
**Blockers**: [None/List issues]
**Next Task**: [From checklist]
EOF

# Edit progress file daily
code progress.md
```

### Metrics Dashboard
```bash
# Create metrics tracker
cat > metrics.json << 'EOF'
{
  "data": {
    "total_samples": 0,
    "recognize": 0,
    "generate": 0,
    "solve": 0,
    "balanced": false
  },
  "dape": {
    "specialized_neurons_pct": 0,
    "key_layers_identified": 0,
    "thinking_vs_direct_diff": 0
  },
  "ablation": {
    "critical_layers_recognize": [],
    "critical_layers_generate": [],
    "critical_layers_solve": [],
    "max_performance_drop": 0
  },
  "probes": {
    "recognize_accuracy": 0,
    "generate_accuracy": 0,
    "solve_accuracy": 0,
    "cross_task_generalization": 0
  },
  "convergence": {
    "methods_agreeing_pct": 0,
    "high_confidence_findings": 0
  }
}
EOF

# Update metrics.json as you complete each analysis
```

---

## 🎯 Success Criteria Checklist

At the end of 6 weeks, you should have:

### Quantitative Results
- [ ] DAPE scores for all 204,800 neurons across 40 layers
- [ ] 5-15% domain-specific neurons identified
- [ ] 3-5 critical layers per ability (all methods agree)
- [ ] Probe accuracies: 65-85% (not trivial)
- [ ] Performance drops: 15-50% when ablating key layers

### Qualitative Insights
- [ ] Can explain which layers handle which abilities
- [ ] Can map concept circuits for maze elements
- [ ] Can identify reasoning-specific neurons
- [ ] Can predict performance from layer patterns
- [ ] Validated hierarchical dependencies (Solve needs R+G)

### Deliverables
- [ ] 20-30 publication-quality figures
- [ ] 4 JSON result files (DAPE, Ablation, Probes, Synthesis)
- [ ] Synthesis report (2,000-3,000 words)
- [ ] 8-10 paper-ready figures with captions
- [ ] Reproducible code + documentation

### Scientific Contributions
- [ ] First interpretability study of Qwen3-14B
- [ ] Neural correlates of spatial reasoning abilities
- [ ] Thinking mode vs direct mode neural comparison
- [ ] Concept circuit maps for maze navigation

---

## 💡 Tips for Success

### Time Management
- **Week 1**: Don't rush data preparation. Good data = good results.
- **Week 2-3**: DAPE is computationally expensive. Run overnight.
- **Week 4**: Linear probes are fast. Do all 8 analyses.
- **Week 5**: Advanced analyses are exploratory. Be creative.
- **Week 6**: Synthesis is crucial. Don't skip this.

### Code Quality
- **Test frequently**: Run with small samples first
- **Save checkpoints**: Save intermediate results
- **Version control**: Commit after each working component
- **Document**: Add comments explaining non-obvious code

### Debugging
- **GPU OOM**: Reduce batch size or use gradient checkpointing
- **Low probe accuracy**: Check label balance
- **High probe accuracy**: Check for task-ability correlation artifact
- **Slow runs**: Use fewer samples for testing

### Communication
- **Weekly updates**: Track progress in `progress.md`
- **Document findings**: Write notes as you discover things
- **Ask questions**: Reference specific sections when stuck

---

## 🔗 File Links

All documentation is in `/home/hokindeng/KnowThat-Neuro/docs/`:

```bash
# Open all docs at once (if your editor supports it)
code \
  docs/QWEN3_INTERPRETABILITY_SUMMARY.md \
  docs/QWEN3_INTERPRETABILITY_PLAN.md \
  docs/QWEN3_QUICK_START_CHECKLIST.md \
  docs/QWEN3_ARCHITECTURE_DIAGRAM.md \
  docs/THREE_MOUNTAIN_INTERPRETABILITY_GUIDE.md
```

---

## 🚦 Next Steps

**Right now, do this:**

1. **Read Summary** (15 min):
   ```bash
   cat docs/QWEN3_INTERPRETABILITY_SUMMARY.md | less
   ```

2. **Skim Plan Sections 1-2** (20 min):
   ```bash
   code docs/QWEN3_INTERPRETABILITY_PLAN.md  # Read sections 1-2
   ```

3. **Check Checklist Phase 1** (10 min):
   ```bash
   cat docs/QWEN3_QUICK_START_CHECKLIST.md | head -300 | less
   ```

4. **Run Prerequisites Check** (5 min):
   ```bash
   # From QWEN3_QUICK_START_CHECKLIST.md Section 1.1
   python --version
   nvidia-smi
   df -h
   ```

5. **If all checks pass, start Week 1 Day 1**:
   ```bash
   # Follow QWEN3_QUICK_START_CHECKLIST.md Phase 1
   ```

---

## 📞 Support

- **Documentation**: All 4 guides in `docs/`
- **Code templates**: QWEN3_INTERPRETABILITY_PLAN.md Section 4
- **Troubleshooting**: QWEN3_QUICK_START_CHECKLIST.md Section 9
- **Methodology**: THREE_MOUNTAIN_INTERPRETABILITY_GUIDE.md

---

**You have everything you need. Time to start! 🚀**

**First action**: `cat docs/QWEN3_INTERPRETABILITY_SUMMARY.md`

---

**Last Updated**: November 2024  
**Status**: Ready to Begin  
**Estimated Completion**: 6 weeks from start  
**Total Documentation**: 4 comprehensive guides (~10,000 lines)

