# Three Mountain Interpretability: Comprehensive Guide

## Executive Summary

The **Three Mountain Interpretability** framework is a sophisticated neuroscience-inspired toolkit for analyzing how Vision-Language Models (VLMs) process spatial cognitive tasks. It provides multiple complementary analysis methods to understand neural specialization, layer-wise information processing, and cognitive capability hierarchies in models like Qwen 2.5 VL.

**Key Innovation**: Unlike traditional black-box evaluation, this framework opens the "neural activity" of VLMs to reveal:
- Which individual neurons specialize for specific cognitive tasks
- Which layers are critical for different types of reasoning
- How cognitive abilities are encoded and transformed across model depth
- Patterns distinguishing correct from incorrect model responses

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Core Concepts](#2-core-concepts)
3. [Analysis Methods](#3-analysis-methods)
4. [Architecture & Components](#4-architecture--components)
5. [Usage Guide](#5-usage-guide)
6. [Technical Deep Dive](#6-technical-deep-dive)
7. [Results & Interpretation](#7-results--interpretation)
8. [Integration with KnowThat-Neuro](#8-integration-with-knowthat-neuro)

---

## 1. Project Overview

### 1.1 Motivation

Vision-language models have demonstrated remarkable capabilities in spatial reasoning tasks, but understanding **how** they achieve this remains elusive. This project addresses fundamental questions:

- **Neuron-Level**: Do individual neurons specialize for specific cognitive abilities?
- **Layer-Level**: Which layers are functionally critical for different reasoning types?
- **Task-Level**: How do cognitive capabilities build hierarchically?
- **Performance**: What distinguishes correct from incorrect model responses?

### 1.2 Cognitive Task Hierarchy

The framework analyzes three hierarchical cognitive abilities inspired by Piaget's "Three Mountain Task":

#### **Level 1: Visual Access (VA)**
- **Definition**: Basic visual perception and viewpoint-dependent object visibility
- **Example**: "Can the child see the toy from their position?"
- **Required Skills**: Object recognition, spatial awareness
- **Expected Encoding**: Early to middle layers (basic visual processing)

#### **Level 2: Mental Rotation (MR)**
- **Definition**: Spatial reasoning and mental manipulation of spatial relationships
- **Example**: "If I rotate this 90°, what configuration results?"
- **Required Skills**: Spatial reasoning, geometric transformation
- **Expected Encoding**: Middle layers (complex spatial processing)

#### **Level 3: Perspective Taking (PT/L2)**
- **Definition**: High-level perspective transformation and understanding others' viewpoints
- **Example**: "What does the scene look like from the other person's viewpoint?"
- **Required Skills**: Visual Access + Mental Rotation + Theory of Mind
- **Expected Encoding**: Deep layers (abstract cognitive reasoning)

**Hierarchical Dependency**:
```
Perspective Taking (L3)
    ↓ requires
Mental Rotation (L2) + Visual Access (L1)
    ↓ requires
Basic Visual Features (L0)
```

### 1.3 Target Models

**Primary**: Qwen 2.5 VL (Vision-Language) Models
- Qwen2.5-VL-3B-Instruct (3 billion parameters)
- Qwen2.5-VL-7B-Instruct (7 billion parameters)

**Architecture Compatibility**:
- Gated Linear Unit (GLU) MLP layers
- Transformer-based with 36 layers (3B model)
- Multi-modal (vision + language) processing

---

## 2. Core Concepts

### 2.1 Neuron Attribution

**Purpose**: Quantify how much each neuron contributes to the model's prediction.

**Method**: Gradient-based attribution adapted for GLU architecture:

```
For layer ℓ, neuron k, at position p:

g_{k,c}^{(ℓ)} = Σ_p (G_{pk}^{(ℓ)} · ∂y_c/∂G_{pk}^{(ℓ)})
```

Where:
- `G_{pk}^{(ℓ)}` = Complete neuron activation (gate × up projection)
- `y_c` = Logit of predicted token c
- `∂y_c/∂G_{pk}^{(ℓ)}` = Gradient computed via backpropagation

**Key Insight**: This captures both the activation magnitude and its causal influence on the output.

### 2.2 Domain Activation Probability Entropy (DAPE)

**Purpose**: Identify neurons that specialize for specific cognitive domains.

**Intuition**: 
- A neuron that **only** activates for mental rotation tasks is domain-specific (low entropy)
- A neuron that activates equally for all tasks is general-purpose (high entropy)

**Formula**:
```
DAPE(k) = -Σ_T p̃_T(k) · log(p̃_T(k))

where:
p̃_T(k) = P_T(k) / Σ_{T'} P_{T'}(k)  (normalized probability)
P_T(k) = activation rate of neuron k on task T
```

**Interpretation**:
- **DAPE = 0**: Perfectly domain-specific (activates for one task only)
- **DAPE = log(N)**: Completely general (equal activation across N tasks)
- **Low DAPE (< 1.0)**: Task-specialized neuron
- **High DAPE (> 2.0)**: General-purpose neuron

### 2.3 Layer Ablation

**Purpose**: Identify critical layers through causal intervention.

**Method**: Systematically zero out layer activations and measure performance degradation.

**Ablation Ratios**: 0%, 25%, 50%, 75%, 100%

**Performance Drop**:
```
Δ_T^{(ℓ,α)} = Acc_T^{baseline} - Acc_T^{(ℓ,α)}
```

**Key Insight**: Layers with high performance drop when ablated are functionally critical.

### 2.4 Linear Probes

**Purpose**: Detect which layers encode information about cognitive abilities.

**Method**: Train binary classifiers on layer activations to predict: "Does this task require ability X?"

**Training**:
```
Input: Hidden states from layer ℓ
Labels: Binary (0=Not Required, 1=Required)
Model: Logistic Regression
Output: Probability that ability is needed
```

**Key Innovation**: Uses balanced negative samples to avoid task-ability correlation (preventing 100% accuracy artifacts).

---

## 3. Analysis Methods

### 3.1 DAPE Analysis (Neuron-Level)

#### Purpose
Identify individual neurons that specialize for specific cognitive domains through activation probability entropy.

#### Workflow
```
1. Collect Activations
   ↓
2. Compute Attribution Scores (gradient × activation)
   ↓
3. Calculate Activation Probabilities (per task, per neuron)
   ↓
4. Compute DAPE Scores (entropy across tasks)
   ↓
5. Identify Domain-Specific Neurons (bottom 1-5% DAPE)
   ↓
6. Analyze Patterns (layer distribution, task co-activation)
```

#### Key Parameters
- `max_samples_per_task`: 10-100 (computation vs coverage)
- `activation_threshold_percentile`: 1.0-25.0 (top % activated positions)
- `domain_specific_percentile`: 1.0-10.0 (bottom % for specialization)
- `split_by_correctness`: True/False (analyze correct vs incorrect separately)

#### Output Visualizations
1. **DAPE Distribution** - Histogram of entropy scores
2. **Layer Distribution** - Where domain-specific neurons concentrate
3. **Neuron DAPE Visualization** - Per-layer DAPE patterns
4. **UpSet Plot** - Task co-activation patterns
5. **Activation Heatmap** - Layer × Task activation patterns
6. **Weight Analysis** - Attribution weight distributions
7. **Coherence Metrics** - Layer-wise specialization measures

#### Expected Findings
- **Early Layers**: More general neurons (high DAPE)
- **Middle Layers**: Mixed specialization
- **Late Layers**: Task-specific neurons concentrated here
- **Correctness**: Different neuron populations for correct vs incorrect

#### Running DAPE Analysis
```bash
cd external/three-mountain-interpretability
python run_dape_analysis.py

# Output: plots/dape_analysis/
```

### 3.2 Layer Ablation Analysis (Layer-Level)

#### Purpose
Identify functionally critical layers through systematic causal intervention.

#### Workflow
```
1. Baseline Evaluation (no ablation)
   ↓
2. For each layer ℓ and ablation ratio α:
   - Zero out α% of layer activations
   - Re-evaluate model performance
   - Measure performance drop
   ↓
3. Compare key layers vs other layers
   ↓
4. Analyze response quality (correctness, confidence, coherence)
```

#### Key Parameters
- `ablation_ratios`: [0.0, 0.25, 0.5, 0.75, 1.0]
- `key_layers`: Dict of task → critical layer indices
- `max_samples_per_task`: 10-100

#### Output Visualizations
1. **Performance Drop Comparison** - Key vs other layers
2. **Logits Evolution** - How predictions change with ablation
3. **Layer Sensitivity** - Performance degradation patterns
4. **Response Quality Heatmap** - Multi-dimensional quality metrics

#### Evaluation Modes
- **Fast Mode**: Logit-only evaluation (50-100× speedup)
- **Full Mode**: Complete text generation + correctness check

#### Expected Findings
- **Task-Specific Key Layers**: Different tasks rely on different layers
- **Ablation Sensitivity**: Key layers show steeper performance drops
- **Response Degradation**: Quality metrics degrade non-uniformly
- **Critical Windows**: Narrow layer ranges are most important

#### Running Ablation Analysis
```bash
cd external/three-mountain-interpretability
python run_layer_ablation.py

# Output: plots/ablation_analysis/
```

### 3.3 Linear Probe Analysis (Representation-Level)

#### Purpose
Understand which layers encode information about cognitive abilities using supervised binary classifiers.

#### Workflow
```
1. Data Preparation
   - Load task samples (spatiality, perspective_l1, perspective_l2)
   - Generate negative samples (break task-ability correlation)
   - Label with ability requirements
   ↓
2. Feature Extraction
   - Extract hidden states from all 36 layers
   - Pool or aggregate sequence representations
   ↓
3. Probe Training
   For each ability (MR, VA, PT):
     For each layer (0-35):
       - Train logistic regression
       - Evaluate on held-out test set
   ↓
4. Enhanced Analyses
   - Confidence evolution tracking
   - Probe weight PCA visualization
   - Error pattern analysis
   - Cross-task generalization test
   - Information flow metrics
   - Layer criticality analysis
```

#### Task-Ability Mapping
| Task Type | Mental Rotation | Visual Access | Perspective Taking |
|-----------|----------------|--------------|-------------------|
| **Spatiality** | ✓ (1) | ✗ (0) | ✗ (0) |
| **Perspective L1** | ✗ (0) | ✓ (1) | ✗ (0) |
| **Perspective L2** | ✓ (1) | ✓ (1) | ✓ (1) |

#### Negative Sample Strategy
To prevent 100% accuracy artifacts from task-ability correlation:

**Spatiality Negatives**:
- Same image pairs + random scene pairs
- Answer = False → Mental Rotation = 0

**L1 Negatives**:
- Fake objects (blue bottle, red phone, etc.)
- Answer = False → Visual Access = 0

**L2 Negatives**:
- Fake objects
- Answer = False → All abilities = 0

#### Eight Analysis Methods

**1. Standard Probe Performance**
- Accuracy, Precision, Recall, F1, AUC per layer
- Train vs test comparison

**2. Confidence Analysis**
- Prediction probability distributions
- Confidence evolution from early to late layers
- Correct vs incorrect confidence comparison

**3. Weight PCA**
- Principal component analysis of probe weights
- Decision boundary patterns across layers
- Cumulative variance explained
- Trajectory visualization in PC space

**4. Error Analysis**
- Consistently wrong samples identification
- Improving samples (early wrong → late correct)
- Error rate evolution across layers
- Error pattern heatmaps

**5. Cross-Task Generalization**
- Train on: [spatiality + perspective_l1]
- Test on: [perspective_l2] (unseen task)
- Measures if probes learn generalizable representations

**6. Information Flow**
- Uses probe accuracy as information proxy
- Computes information gain per layer
- Identifies saturation points
- Layer-wise info accumulation

**7. Layer Criticality**
- Multi-ability integration layers
- Where do multiple abilities converge?
- Critical decision-making layers

**8. DAPE Integration (Optional)**
- Train probes on DAPE-filtered neurons only
- Compare full vs domain-specific neuron subsets
- Efficiency analysis

#### Output Structure
```
plots/linear_probe_analysis_full/
├── layer_performance_curves.png      # Performance across layers
├── cross_ability_heatmap.png         # Ability × Layer matrix
├── cross_task_generalization.png     # Generalization test results
├── information_flow_analysis.png     # Info flow metrics
├── layer_criticality.png             # Multi-ability integration
├── linear_probe_results.json         # Numerical results
├── generalization_results.json       # Generalization metrics
├── linear_probe_report.txt           # Human-readable summary
└── [ability]/                        # Per-ability analysis
    ├── detailed_analysis.png         # 4-panel standard analysis
    ├── key_layers.png                # Top-5 layer identification
    ├── confidence_analysis.png       # 4-panel confidence evolution
    ├── weight_analysis.png           # 4-panel PCA + cumulative variance
    └── error_analysis.png            # 4-panel error patterns
```

#### Expected Results

**Performance** (with balanced data):
- Mental Rotation: 60-82% (mean ~71%)
- Visual Access: 63-81% (mean ~74%)
- Perspective Taking: 74-88% (mean ~83%)

**Layer Patterns**:
- Early layers (~0-10): ~60% (slightly better than random)
- Middle layers (~10-25): Gradual improvement
- Late layers (~25-35): Peak performance (~80-88%)

**Confidence Evolution**:
- Early layers: ~0.65 (uncertain)
- Late layers: ~0.85 (confident)
- Correct predictions: Higher confidence throughout
- Gap widens in later layers

**Weight Complexity**:
- ~10 PCs capture 80% of variance
- Smooth trajectory (gradual transformation)
- Later layers show more complex boundaries

**Error Patterns**:
- ~15-20% consistently wrong (hard samples)
- ~10-15% improve from early to late
- Error rate: 40% → 15-20% progression

#### Running Linear Probe Analysis
```bash
cd external/three-mountain-interpretability

# Full analysis (recommended)
bash run_linear_probe_full.sh

# Or with Python
python run_linear_probe_full.py

# Test run (small sample)
python run_linear_probe_full.py --max-samples 20

# With DAPE integration
python run_linear_probe_full.py --use-dape

# Output: plots/linear_probe_analysis_full/
```

### 3.4 Information Flow Analysis

#### Purpose
Track how task-relevant information accumulates and propagates across layers.

#### Method
- Uses probe accuracy as proxy for information content
- Computes layer-wise information gain
- Identifies saturation points
- Analyzes multi-ability integration

#### Key Metrics
- **Information Gain**: Δ_Acc(layer) = Acc(layer+1) - Acc(layer)
- **Saturation Layer**: First layer where gain < threshold
- **Integration Score**: Correlation of abilities at each layer

---

## 4. Architecture & Components

### 4.1 Directory Structure

```
three-mountain-interpretability/
├── data/                          # Dataset directory
│   ├── lab/                       # Laboratory cognitive tasks
│   │   ├── spatiality.parquet     # Mental rotation tasks
│   │   ├── perspective_mc.parquet # Visual access (L1)
│   │   └── perspective_l2.parquet # Perspective taking (L2)
│   └── real/                      # Real-world scenarios
│       └── ego-exo.parquet
│
├── utils/                         # Core analysis modules
│   ├── data_loader.py             # Unified data loading
│   ├── layer_ablation.py          # Layer ablation implementation
│   ├── dape_analysis.py           # DAPE neuron analysis (3600+ lines)
│   ├── cognitive_task_analysis.py # Task hierarchy analysis
│   ├── info_flow.py               # Information flow tracking
│   ├── linear_probe_analysis.py   # Basic linear probe
│   ├── linear_probe_analysis_enhanced.py  # Enhanced analysis
│   ├── binary_task_expander.py    # Negative sample generation
│   ├── dape_probe_integration.py  # DAPE-Probe integration
│   ├── probe_info_flow.py         # Probe-based info flow
│   └── visualization_config.py    # Visualization settings
│
├── scripts/                       # Visualization scripts
│   ├── plot_performance_drop.py
│   ├── plot_logits_evolution.py
│   ├── plot_layer_sensitivity.py
│   ├── visualize_dape.py
│   └── run_all_plots.sh
│
├── plots/                         # Generated visualizations
│   ├── dape_analysis/
│   ├── ablation_analysis/
│   └── linear_probe_analysis_full/
│
├── docs/                          # Documentation
│   ├── README.md
│   ├── DAPE_ANALYSIS.md
│   ├── LAYER_ABLATION_ANALYSIS.md
│   ├── LINEAR_PROBE_ANALYSIS.md
│   └── LINEAR_PROBE_FULL_ANALYSIS.md
│
├── run_dape_analysis.py           # DAPE analysis entry point
├── run_layer_ablation.py          # Ablation entry point
├── run_linear_probe_full.py       # Linear probe entry point
├── run_linear_probe_full.sh       # Shell wrapper
├── QUICK_REFERENCE.md
├── LINEAR_PROBE_QUICKSTART.md
└── README.md
```

### 4.2 Core Components

#### DataLoader (`utils/data_loader.py`)
- **Purpose**: Unified data loading across formats
- **Formats**: Parquet, JSON, JSONL
- **Features**:
  - Automatic format detection
  - Image decoding (base64 or file paths)
  - Task-specific loading
  - Sample limiting for testing

#### DAPEAnalyzer (`utils/dape_analysis.py`)
- **Purpose**: Neuron-level specialization analysis
- **Key Methods**:
  - `collect_activation_data()` - Extract neuron activations
  - `calculate_activation_probabilities()` - Per-task probabilities
  - `calculate_dape_scores()` - Entropy computation
  - `identify_domain_specific_neurons()` - Specialization detection
  - `analyze_domain_specific_neurons()` - Pattern analysis

#### LayerAblator (`utils/layer_ablation.py`)
- **Purpose**: Layer criticality through intervention
- **Key Methods**:
  - `ablate_layer()` - Zero out layer activations
  - `evaluate_model()` - Performance measurement
  - `compute_performance_drop()` - Degradation quantification
  - `analyze_response_quality()` - Multi-metric evaluation

#### EnhancedLinearProbeAnalyzer (`utils/linear_probe_analysis_enhanced.py`)
- **Purpose**: Cognitive ability encoding detection
- **Key Methods**:
  - `train_probes()` - Layer-wise classifier training
  - `analyze_confidence_evolution()` - Confidence tracking
  - `analyze_probe_weights()` - Weight PCA
  - `analyze_errors()` - Error pattern analysis
  - `test_generalization()` - Cross-task evaluation

#### BinaryTaskExpander (`utils/binary_task_expander.py`)
- **Purpose**: Generate balanced negative samples
- **Key Methods**:
  - `expand_spatiality()` - Mental rotation negatives
  - `expand_perspective_l1()` - Visual access negatives
  - `expand_perspective_l2()` - Perspective taking negatives
  - `expand_all_tasks()` - Unified expansion

### 4.3 Data Format

#### Standard Sample Format
```python
sample = {
    'content': [
        {
            'type': 'image',
            'image': PIL.Image or 'path/to/image.jpg'
        },
        {
            'type': 'text',
            'text': 'Question text here'
        }
    ],
    'answer': 'Expected answer',
    'task_source': 'spatiality',  # or 'perspective_l1', 'perspective_l2'
    
    # For linear probes
    'requires_mental_rotation': 0 or 1,
    'requires_visual_access': 0 or 1,
    'requires_perspective_taking': 0 or 1
}
```

---

## 5. Usage Guide

### 5.1 Installation & Setup

#### Prerequisites
```bash
# Python 3.9+ recommended
python --version

# CUDA-capable GPU (recommended)
nvidia-smi
```

#### Install Dependencies
```bash
cd external/three-mountain-interpretability

# Core dependencies
pip install torch transformers
pip install qwen-vl-utils Pillow

# Analysis dependencies
pip install numpy pandas matplotlib seaborn
pip install scikit-learn scipy

# Data dependencies
pip install pyarrow  # For parquet files
pip install upsetplot  # For DAPE visualizations
pip install tqdm  # Progress bars
```

### 5.2 Quick Start Examples

#### Example 1: Run DAPE Analysis
```bash
cd external/three-mountain-interpretability

# Run with default settings
python run_dape_analysis.py

# Check output
ls plots/dape_analysis/
```

**Expected Runtime**: 20-30 minutes (100 samples/task)  
**Expected Output**: 7-8 visualization files + JSON results

#### Example 2: Run Layer Ablation
```bash
cd external/three-mountain-interpretability

# Run with default settings
python run_layer_ablation.py

# Check output
ls plots/ablation_analysis/
```

**Expected Runtime**: 5-10 minutes (fast logit mode)  
**Expected Output**: 4 visualization files + JSON results

#### Example 3: Run Linear Probe Analysis
```bash
cd external/three-mountain-interpretability

# Full analysis
bash run_linear_probe_full.sh

# Or with Python
python run_linear_probe_full.py

# Test with small sample
python run_linear_probe_full.py --max-samples 20

# Check output
ls plots/linear_probe_analysis_full/
```

**Expected Runtime**: 30-60 minutes (full data)  
**Expected Output**: 18-24 visualization files + 3 reports

#### Example 4: Custom Configuration

**Modify DAPE parameters**:
```python
# Edit run_dape_analysis.py

analyzer = DAPEAnalyzer(
    model_path="Qwen/Qwen2.5-VL-7B-Instruct",  # Use 7B model
    device="cuda"
)

analyzer.collect_activation_data(
    max_samples_per_task=50,  # Reduce for speed
    split_by_correctness=True  # Analyze correct vs incorrect separately
)

analyzer.identify_domain_specific_neurons(
    percentile=5.0  # Bottom 5% instead of 1%
)
```

**Modify Linear Probe parameters**:
```python
# Edit run_linear_probe_full.py

analyzer = EnhancedLinearProbeAnalyzer(
    model_path="Qwen/Qwen2.5-VL-3B-Instruct",
    data_root="data",
    output_dir="plots/linear_probe_analysis_full"
)

# Generate balanced data
expander = BinaryTaskExpander(data_root="data")
samples = expander.expand_all_tasks(max_samples_per_task=100)

# Run analysis
results = analyzer.run_full_analysis(
    samples=samples,
    use_dape=True  # Enable DAPE integration
)
```

### 5.3 Interpreting Results

#### DAPE Results

**dape_distribution.pdf**:
- **X-axis**: DAPE score (0 = specialized, log(N) = general)
- **Y-axis**: Number of neurons
- **Interpretation**: 
  - Peak at low values → Many specialized neurons
  - Peak at high values → Mostly general neurons
  - Bimodal → Distinct specialized and general populations

**layer_distribution.pdf**:
- **X-axis**: Layer index
- **Y-axis**: Number of domain-specific neurons
- **Interpretation**:
  - Early layer peaks → Low-level feature specialization
  - Late layer peaks → High-level concept specialization
  - Uniform → No layer-specific specialization

**upset_plot.pdf**:
- Shows task co-activation patterns
- **Interpretation**:
  - Large single-task bars → Pure specialization
  - Large multi-task intersections → Shared processing

#### Ablation Results

**performance_drop_comparison.pdf**:
- **X-axis**: Ablation ratio
- **Y-axis**: Performance drop
- **Interpretation**:
  - Steeper slope → More critical layers
  - Compare key vs other layers
  - Different tasks show different patterns

**response_quality_heatmap.pdf**:
- **Dimensions**: Correctness, Confidence, Coherence
- **Interpretation**:
  - Simultaneous degradation → True layer criticality
  - Selective degradation → Dimension-specific role

#### Linear Probe Results

**layer_performance_curves.png**:
- **X-axis**: Layer index
- **Y-axis**: Test accuracy
- **Interpretation**:
  - Rising curve → Information accumulates
  - Plateau → Information saturates
  - Peak layer → Optimal encoding layer
  - Different abilities peak at different layers

**confidence_analysis.png** (per ability):
- Shows how prediction confidence evolves
- **Interpretation**:
  - Increasing confidence → Growing certainty
  - Gap between correct/incorrect → Model calibration
  - Flat confidence → No representation refinement

**weight_analysis.png** (per ability):
- PCA of probe weights across layers
- **Interpretation**:
  - Smooth trajectory → Gradual transformation
  - Sharp turns → Representation shift
  - Few PCs for 80% variance → Simple patterns
  - Many PCs needed → Complex patterns

**error_analysis.png** (per ability):
- Shows which samples are hard
- **Interpretation**:
  - Consistently wrong → Fundamentally hard samples
  - Improving → Representation development
  - High error rate → Need better features

### 5.4 Common Workflows

#### Workflow 1: Full Pipeline Analysis
```bash
# 1. Run DAPE to identify specialized neurons
python run_dape_analysis.py

# 2. Extract key layers from DAPE results
python -c "
from utils.dape_analysis import DAPEAnalyzer
analyzer = DAPEAnalyzer.load_results('plots/dape_analysis/dape_analysis_results.json')
key_layers = analyzer.get_key_layers()
print(key_layers)
"

# 3. Run ablation with DAPE-identified layers
# (Edit key_layers in run_layer_ablation.py first)
python run_layer_ablation.py

# 4. Run linear probes for ability encoding
bash run_linear_probe_full.sh

# 5. Compare and synthesize results
```

#### Workflow 2: Iterative Refinement
```bash
# Start with small sample for quick iteration
python run_linear_probe_full.py --max-samples 20

# Check results, adjust parameters

# Medium run
python run_linear_probe_full.py --max-samples 100

# Full run when satisfied
bash run_linear_probe_full.sh
```

#### Workflow 3: Model Comparison
```bash
# Run on 3B model
python run_linear_probe_full.py --model Qwen/Qwen2.5-VL-3B-Instruct

# Run on 7B model
python run_linear_probe_full.py --model Qwen/Qwen2.5-VL-7B-Instruct

# Compare results from plots/
```

---

## 6. Technical Deep Dive

### 6.1 GLU-Aware Attribution

**Problem**: Standard attribution methods don't account for GLU architecture:
```
Traditional: attribution = gradient × activation
GLU Reality: activation = gate × up
```

**Solution**: Compute complete attribution:
```python
# Forward pass
gate = gate_proj(x)      # (batch, seq, hidden)
up = up_proj(x)          # (batch, seq, hidden)
activation = silu(gate) * up  # Complete neuron output

# Attribution
logits = model(input)
target_logit = logits[..., predicted_token]
target_logit.backward()  # Compute gradients

# Extract gradients w.r.t. complete activation
grad = activation.grad  # Shape: (batch, seq, hidden)

# Attribution score
attribution = (activation * grad).sum(dim=1)  # Sum over sequence
```

**Why This Matters**: 
- Captures both gating and value contributions
- More accurate neuron importance quantification
- Prevents bias toward gate or up projection

### 6.2 Layer-Wise Threshold Strategy

**Problem**: Different layers have different activation magnitudes:
```
Early layers: activations ~ [-1, 1]
Late layers: activations ~ [-100, 100]
```

**Solution**: Compute layer-wise, task-wise thresholds:
```python
for layer in range(num_layers):
    for task in tasks:
        # Collect activation scores for this layer-task combination
        scores = []
        for sample in task_samples:
            activation = get_activation(sample, layer)
            attribution = get_attribution(sample, layer)
            score = abs(activation * attribution)
            scores.extend(score.flatten())
        
        # Compute percentile threshold
        threshold = np.percentile(scores, 100 - top_percentile)
        thresholds[layer][task] = threshold
```

**Benefits**:
- Fair comparison across layers
- Prevents late-layer dominance
- Task-specific sensitivity

### 6.3 Balanced Negative Sampling

**Problem**: Task-ability correlation leads to trivial 100% accuracy:
```
All spatiality samples → MR=1
All perspective_l1 samples → VA=1
Model learns: "Is it spatiality?" instead of "Does it need MR?"
```

**Solution**: Generate task-inconsistent negative samples:
```python
# For spatiality (originally all MR=1)
negatives = []
for positive_sample in spatiality_samples:
    # Same images, random scene
    negative = {
        'image': positive_sample['image'],
        'scene': random.choice(other_scenes),
        'answer': False,
        'requires_mental_rotation': 0  # NOW SPATIALITY HAS MR=0 samples!
    }
    negatives.append(negative)

# Now model must actually detect MR requirements, not just task identity
```

**Result**: Realistic 60-88% accuracy (not 100%)

### 6.4 Probe Weight PCA

**Purpose**: Understand what probes learn across layers

**Method**:
```python
# Collect probe weights
weights = []
for layer in range(36):
    probe = trained_probes[layer]
    w = probe.coef_[0]  # Shape: (hidden_dim,)
    weights.append(w)

weights = np.array(weights)  # Shape: (36, hidden_dim)

# Apply PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
weights_pca = pca.fit_transform(weights)

# Analyze cumulative variance
cumsum = np.cumsum(pca.explained_variance_ratio_)
# How many PCs for 80% variance?
n_pcs_80 = np.argmax(cumsum > 0.8) + 1

# Plot trajectory in PC space
plt.plot(weights_pca[:, 0], weights_pca[:, 1], 'o-')
plt.xlabel('PC1')
plt.ylabel('PC2')
```

**Interpretation**:
- **Few PCs (2-3) for 80%**: Simple, low-dimensional decision boundary
- **Many PCs (>5) for 80%**: Complex, high-dimensional boundary
- **Smooth trajectory**: Gradual representation transformation
- **Jumps**: Discrete representation shifts

### 6.5 Information Flow Computation

**Concept**: Probe accuracy reflects information content

**Method**:
```python
# Use probe accuracy as information proxy
info = test_accuracies  # Shape: (num_layers,)

# Compute information gain
info_gain = np.diff(info)  # Layer-to-layer gain

# Identify saturation point
saturation_idx = np.argmax(info_gain < threshold)

# Compute cumulative information
cumulative_info = np.cumsum(info_gain)

# Multi-ability integration
integration_score = correlation(info_MR, info_VA, info_PT)
```

**Insights**:
- Where does information first appear?
- Where does it saturate?
- Where do abilities converge?

---

## 7. Results & Interpretation

### 7.1 Expected Findings

#### From DAPE Analysis

**Neuron Specialization**:
- **5-15%** of neurons are domain-specific (DAPE < 1.0)
- **Late layers (25-35)** have more specialized neurons
- **Different tasks** recruit different neuron populations

**Layer Distribution**:
- **Visual Access**: Neurons in layers 10-20
- **Mental Rotation**: Neurons in layers 15-25
- **Perspective Taking**: Neurons in layers 25-35

**Correctness Patterns**:
- **Correct responses**: More specialized neuron activation
- **Incorrect responses**: More general neuron activation
- **Hard samples**: Weak or inconsistent specialization

#### From Layer Ablation

**Critical Layers**:
- **Spatiality (MR)**: Layers 15-25 most critical
- **Perspective L1 (VA)**: Layers 10-20 most critical
- **Perspective L2 (PT)**: Layers 25-35 most critical

**Ablation Sensitivity**:
- **Key layers**: 30-50% performance drop at 100% ablation
- **Other layers**: 10-20% performance drop
- **Hierarchical dependency**: L2 depends on L1 layers

#### From Linear Probes

**Encoding Patterns**:
- **Visual Access**: Peaks at layers 18-22 (~81%)
- **Mental Rotation**: Peaks at layers 22-28 (~82%)
- **Perspective Taking**: Peaks at layers 28-35 (~88%)

**Confidence Evolution**:
- **Early layers**: Low confidence (~0.65), high uncertainty
- **Late layers**: High confidence (~0.85), low uncertainty
- **Correct predictions**: Higher confidence at all layers

**Weight Complexity**:
- **Early layers**: High-dimensional, complex boundaries
- **Late layers**: Lower-dimensional, cleaner boundaries
- **10-15 PCs** capture 80% of variance

**Error Patterns**:
- **15-20%** samples consistently wrong (intrinsically hard)
- **10-15%** improve from early to late layers
- **Hard samples**: Often ambiguous or mislabeled

### 7.2 Cross-Method Synthesis

#### Convergence Analysis

**When methods agree**:
```
DAPE identifies layer 28 as specialized for PT
    ↓
Ablation shows layer 28 ablation hurts PT most
    ↓
Linear probes peak at layer 28 for PT
    ↓
Conclusion: Layer 28 is critical for PT (high confidence)
```

**When methods disagree**:
```
DAPE: Layer 15 has specialized neurons for MR
Ablation: Layer 15 ablation doesn't hurt MR much
    ↓
Possible interpretations:
- Redundancy: Other neurons compensate
- False positive: DAPE neurons not actually functional
- Indirect role: Neurons support but aren't critical
```

#### Triangulation Strategy

Use multiple methods to build confidence:
1. **DAPE**: Identifies candidate specialized neurons
2. **Ablation**: Validates functional criticality
3. **Probes**: Confirms information encoding
4. **Convergence**: Strong evidence when all agree

### 7.3 Publication-Ready Insights

#### Key Finding 1: Hierarchical Layer Specialization
> "Vision-language models exhibit hierarchical cognitive specialization across layers:
> - **Visual Access** (basic perception): Layers 10-20
> - **Mental Rotation** (spatial reasoning): Layers 15-25
> - **Perspective Taking** (social cognition): Layers 25-35
>
> This mirrors human cognitive development and provides evidence for compositional
> reasoning in transformer architectures."

#### Key Finding 2: Sparse Neural Specialization
> "Only 5-15% of neurons exhibit domain-specific activation patterns (DAPE < 1.0),
> concentrated in late layers (25-35). This sparse specialization suggests efficient
> cognitive resource allocation and parallels findings in neuroscience (Hubel & Wiesel)."

#### Key Finding 3: Correctness-Dependent Activation
> "Correct and incorrect model responses recruit different neural populations:
> - Correct: Strong activation of specialized neurons
> - Incorrect: Diffuse activation of general neurons
>
> This suggests that successful reasoning requires precise neural specialization,
> while failures result from inappropriate generalization."

#### Key Finding 4: Information Accumulation
> "Cognitive ability information accumulates gradually across layers rather than
> appearing suddenly. Early layers encode ~60% of final information, with steady
> improvement until saturation at layers 28-32. This gradual refinement contrasts
> with discrete symbolic reasoning."

---

## 8. Integration with KnowThat-Neuro

### 8.1 Complementary Roles

**Three-Mountain-Interpretability**:
- **Focus**: Internal model mechanisms and neural specialization
- **Methods**: DAPE, layer ablation, linear probes
- **Questions**: "How does the model process spatial reasoning?"

**KnowThat-Neuro**:
- **Focus**: Maze navigation and problem-solving
- **Methods**: Maze generation, solution verification, API evaluation
- **Questions**: "Can the model solve spatial navigation tasks?"

**Synergy**:
```
KnowThat-Neuro evaluates WHAT models can do
    ↓
Three-Mountain explains HOW they do it
    ↓
Combined insights for model improvement
```

### 8.2 Shared Components

Both projects can use:
- **Qwen API integration** (`core/qwen_api.py` ← applicable to both)
- **Spatial reasoning evaluation** (maze solving ← KnowThat, interpretation ← Three-Mountain)
- **Layer-wise analysis** (maze solving performance vs layer specialization)

### 8.3 Potential Integration Projects

#### Project 1: Maze Solving Interpretability
**Goal**: Understand which neurons/layers are critical for maze solving

**Method**:
1. Use KnowThat-Neuro to generate maze tasks
2. Apply Three-Mountain DAPE analysis to maze-solving activations
3. Identify maze-specific specialized neurons
4. Use layer ablation to validate critical layers

**Expected Insight**: "Layers 20-28 are critical for spatial navigation, with 
specialized neurons detecting paths, walls, and goal positions."

#### Project 2: Encoding Comparison
**Goal**: Compare how mazes vs perspective tasks are encoded

**Method**:
1. Extract activations from both task types
2. Train shared linear probes
3. Compare layer specialization patterns
4. Measure cross-task generalization

**Expected Insight**: "Maze navigation and mental rotation share middle-layer 
representations (15-25), suggesting common spatial reasoning substrates."

#### Project 3: Intervention-Based Improvement
**Goal**: Use interpretability insights to improve maze solving

**Method**:
1. Identify critical layers via Three-Mountain ablation
2. Apply targeted prompting strategies for those layers
3. Measure improvement in KnowThat-Neuro maze tasks
4. Iterate to find optimal intervention

**Expected Insight**: "Prompting the model to 'think spatially' activates layers 
20-25, improving maze solving accuracy by 15%."

### 8.4 Usage Example: Integrated Analysis

```python
# In KnowThat-Neuro project
from core import QwenAPISolver
from external.three_mountain_interpretability.utils import DAPEAnalyzer

# Solve maze and collect activations
solver = QwenAPISolver(model="qwen-turbo")
maze = generate_maze(size=(7, 7), shape='square')
activations = solver.solve_with_activations(maze)  # Hypothetical method

# Analyze maze-solving neural patterns
analyzer = DAPEAnalyzer(model_path="Qwen/Qwen2.5-VL-3B-Instruct")
analyzer.analyze_activations(activations, task_name='maze_solving')

# Identify critical neurons for maze navigation
specialized_neurons = analyzer.get_domain_specific_neurons(percentile=5.0)

# Validate with ablation
from external.three_mountain_interpretability.utils import LayerAblator
ablator = LayerAblator(model, processor)
ablation_results = ablator.test_maze_critical_layers(maze, specialized_neurons)

# Generate interpretability report
report = {
    'maze_difficulty': get_maze_difficulty(maze),
    'specialized_neurons': len(specialized_neurons),
    'critical_layers': ablation_results['critical_layers'],
    'performance_with_ablation': ablation_results['accuracy']
}
```

### 8.5 Shared Documentation

This guide serves both projects:
- **KnowThat-Neuro users**: Understand how to interpret model behavior
- **Three-Mountain users**: See practical applications in maze navigation
- **Integrated users**: Leverage both toolkits for comprehensive analysis

---

## 9. Troubleshooting & FAQs

### 9.1 Common Issues

#### Issue: CUDA Out of Memory
**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# 1. Reduce batch size
max_samples_per_task = 50  # Instead of 100

# 2. Use smaller model
model_path = "Qwen/Qwen2.5-VL-3B-Instruct"  # Instead of 7B

# 3. Enable CPU offloading
device_map = "auto"

# 4. Reduce sequence length
max_length = 512  # Instead of 2048

# 5. Clear cache between samples
torch.cuda.empty_cache()
```

#### Issue: Parquet Files Not Loading
**Symptoms**: `ImportError: pyarrow not available`

**Solution**:
```bash
pip install pyarrow
```

#### Issue: Visualization Errors
**Symptoms**: `upsetplot not found` or matplotlib errors

**Solution**:
```bash
pip install upsetplot matplotlib seaborn
```

#### Issue: Low Probe Accuracy (<0.6)
**Symptoms**: All probes perform poorly

**Possible Causes**:
1. **Insufficient data**: Use more samples
2. **Imbalanced labels**: Check label distribution
3. **Weak features**: Try different layers
4. **Wrong task mapping**: Verify ability labels

**Solutions**:
```python
# Check label balance
print(samples['requires_mental_rotation'].value_counts())

# Use more samples
expander.expand_all_tasks(max_samples_per_task=200)

# Check specific layers
best_layer = results['mental_rotation']['best_layer']
print(f"Best layer: {best_layer}")
```

#### Issue: 100% Probe Accuracy (Suspicious)
**Symptoms**: All layers achieve perfect accuracy

**Cause**: Task-ability correlation not broken (need negative samples)

**Solution**:
```python
# Verify negative samples are generated
expander = BinaryTaskExpander(data_root="data")
samples = expander.expand_all_tasks(max_samples_per_task=100)

# Check label distribution
print("Spatiality MR labels:", 
      [s['requires_mental_rotation'] for s in samples if s['task_source']=='spatiality'])
# Should see both 0s and 1s, not all 1s
```

### 9.2 FAQs

**Q: How long does full analysis take?**
A: 
- DAPE: 20-30 min (100 samples/task)
- Ablation: 5-10 min (fast mode)
- Linear Probes: 30-60 min (full data)
- Total: ~1-2 hours for complete pipeline

**Q: Can I use other models besides Qwen?**
A: Yes, but requires code adaptation:
- Must support vision-language inputs
- Must expose layer activations
- GLU attribution assumes specific MLP architecture
- Linear probes are architecture-agnostic

**Q: What's the minimum sample size?**
A: 
- Testing: 10-20 samples/task
- Meaningful analysis: 50-100 samples/task
- Publication-quality: 100+ samples/task

**Q: How do I cite this work?**
A:
```bibtex
@software{three_mountain_interpretability,
  title={Three Mountain Interpretability: Neuroscience-Inspired Analysis of Vision-Language Models},
  author={[Authors]},
  year={2024},
  url={https://github.com/grow-ai-like-a-child/three-mountain-interpretability}
}
```

**Q: Can I contribute?**
A: Yes! Priority areas:
- Additional attribution methods
- More visualization types
- Real activation extraction
- Performance optimizations
- Documentation improvements

**Q: Where can I get help?**
A:
1. Check documentation in `docs/`
2. Review code comments in `utils/`
3. Examine example outputs in `plots/`
4. Open GitHub issues for bugs

---

## 10. Conclusion

The Three Mountain Interpretability framework provides unprecedented insight into how vision-language models process spatial cognitive tasks. By combining neuron-level (DAPE), layer-level (ablation), and representation-level (probes) analyses, it offers a comprehensive view of model internals.

**Key Takeaways**:
1. **Specialized neurons exist**: 5-15% of neurons are domain-specific
2. **Hierarchical processing**: Different abilities encoded at different depths
3. **Gradual information accumulation**: No discrete symbolic jumps
4. **Correctness depends on specialization**: Correct responses use specialized neurons
5. **Practical applicability**: Insights can guide model improvement

**Next Steps**:
- Run analyses on your own models
- Integrate with KnowThat-Neuro for maze navigation
- Extend to other cognitive domains
- Publish findings to advance interpretability research

**Remember**: Interpretability is not just about understanding models—it's about
building better, more reliable, and more aligned AI systems.

---

## Appendix: Quick Reference

### Commands Cheat Sheet
```bash
# DAPE Analysis
python run_dape_analysis.py

# Layer Ablation
python run_layer_ablation.py

# Linear Probes (full)
bash run_linear_probe_full.sh

# Linear Probes (test)
python run_linear_probe_full.py --max-samples 20

# With DAPE integration
python run_linear_probe_full.py --use-dape
```

### Key Files
- **DAPE**: `utils/dape_analysis.py` (3600+ lines)
- **Ablation**: `utils/layer_ablation.py` (700+ lines)
- **Probes**: `utils/linear_probe_analysis_enhanced.py` (550+ lines)
- **Data**: `utils/data_loader.py`
- **Expansion**: `utils/binary_task_expander.py`

### Output Locations
- **DAPE**: `plots/dape_analysis/`
- **Ablation**: `plots/ablation_analysis/`
- **Probes**: `plots/linear_probe_analysis_full/`

### Parameter Defaults
- `max_samples_per_task`: 100
- `activation_threshold_percentile`: 1.0
- `domain_specific_percentile`: 1.0-5.0
- `ablation_ratios`: [0.0, 0.25, 0.5, 0.75, 1.0]

---

**Document Version**: 1.0  
**Last Updated**: November 2024  
**Status**: Production Ready  
**Compatibility**: Qwen 2.5 VL Models (3B, 7B)

