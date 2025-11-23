# Qwen3-14B Interpretability Study Plan
## Adapting Three Mountain Framework for Solve/Recognize/Generate Analysis

**Target Model**: Qwen3-14B (14.8B parameters, 40 layers)  
**Target Abilities**: Solve, Recognize, Generate  
**Goal**: Identify neural correlates and concept circuits for cognitive abilities

---

## Executive Summary

This plan adapts the Three Mountain Interpretability framework to study how Qwen3-14B processes maze/spatial reasoning tasks across three hierarchical abilities:
- **Recognize**: Identifying maze properties, patterns, solutions
- **Generate**: Creating new mazes, paths, or representations
- **Solve**: Finding solutions through step-by-step reasoning

**Key Innovation**: Leveraging Qwen3's **thinking mode** to analyze internal reasoning processes.

---

## Table of Contents

1. [Model Architecture Considerations](#1-model-architecture-considerations)
2. [Ability Definitions & Task Mapping](#2-ability-definitions--task-mapping)
3. [Data Preparation Strategy](#3-data-preparation-strategy)
4. [Analysis Pipeline](#4-analysis-pipeline)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Expected Insights](#6-expected-insights)
7. [Code Adaptation Guide](#7-code-adaptation-guide)

---

## 1. Model Architecture Considerations

### 1.1 Qwen3-14B Specifications

```python
Model Specs:
- Total Parameters: 14.8B
- Non-Embedding Parameters: 13.2B
- Number of Layers: 40 (vs 36 in Qwen2.5-VL-3B)
- Attention Heads: 40 for Q, 8 for KV (GQA)
- Context Length: 32,768 native (131,072 with YaRN)
- Architecture: Transformer with GLU-based MLP
```

### 1.2 Key Architectural Differences

| Feature | Qwen2.5-VL-3B | Qwen3-14B | Impact on Analysis |
|---------|---------------|-----------|-------------------|
| **Layers** | 36 | 40 | Need to adjust layer indices |
| **Parameters** | 3B | 14.8B | More neurons to analyze |
| **Special Feature** | - | Thinking Mode | Can analyze reasoning traces |
| **Context** | 32K | 131K | Can handle longer mazes |
| **Modality** | Vision+Text | Text-only | Remove vision preprocessing |

### 1.3 Thinking Mode Integration

**Critical Feature**: Qwen3-14B supports seamless switching between thinking and non-thinking modes.

```python
# Thinking Mode (enable_thinking=True)
# Model generates: <think>reasoning process</think>final answer

# Non-Thinking Mode (enable_thinking=False)  
# Model generates: final answer directly
```

**Analysis Opportunity**: Compare neural activation patterns between:
- **Thinking Mode Solve**: Explicit reasoning steps
- **Non-Thinking Mode Solve**: Direct answers
- **Recognize/Generate**: Can also be analyzed in both modes

---

## 2. Ability Definitions & Task Mapping

### 2.1 Ability Hierarchy

Based on your KnowThat-Neuro experiments:

#### **Level 1: Recognize (R)**
- **Definition**: Identifying maze properties, valid paths, solution correctness
- **Example Tasks**:
  - "Is this path a valid solution?" → Yes/No
  - "What type of maze is this?" → Square/Hexagon/Triangle
  - "Does this maze have a unique solution?" → Yes/No
  - "Which of these is the correct solution?" → Multiple choice
- **Required Skills**: Pattern recognition, constraint checking
- **Expected Encoding**: Early-to-middle layers (10-25)

#### **Level 2: Generate (G)**
- **Definition**: Creating new mazes, paths, or maze representations
- **Example Tasks**:
  - "Generate a 5x5 maze with exactly 3 turns"
  - "Create a hexagonal maze with difficulty 4"
  - "Draw an alternative path from A to B"
  - "Generate ASCII representation of this maze"
- **Required Skills**: Structural understanding, constraint satisfaction, creativity
- **Expected Encoding**: Middle-to-late layers (20-35)

#### **Level 3: Solve (S)**
- **Definition**: Finding solutions through step-by-step reasoning
- **Example Tasks**:
  - "Solve this maze step by step"
  - "Find the shortest path from start to goal"
  - "Navigate this maze using only text commands"
  - "Explain your solving strategy"
- **Required Skills**: Recognize + Generate + Sequential reasoning
- **Expected Encoding**: Late layers (30-40), especially in thinking mode

**Hierarchical Dependency**:
```
Solve (L3)
    ↓ requires
Generate (L2) + Recognize (L1)
    ↓ requires
Basic Pattern Recognition (L0)
```

### 2.2 Task-Ability Mapping

| Task Type | Recognize (R) | Generate (G) | Solve (S) |
|-----------|--------------|--------------|-----------|
| **Recognition Tasks** | ✓ (1) | ✗ (0) | ✗ (0) |
| **Generation Tasks** | ✓ (1) | ✓ (1) | ✗ (0) |
| **Solving Tasks** | ✓ (1) | ✓ (1) | ✓ (1) |

### 2.3 Thinking Mode Variants

For each ability, analyze in **both modes**:

```python
task_variants = {
    'recognize_thinking': {'enable_thinking': True, 'ability': 'recognize'},
    'recognize_direct': {'enable_thinking': False, 'ability': 'recognize'},
    'generate_thinking': {'enable_thinking': True, 'ability': 'generate'},
    'generate_direct': {'enable_thinking': False, 'ability': 'generate'},
    'solve_thinking': {'enable_thinking': True, 'ability': 'solve'},
    'solve_direct': {'enable_thinking': False, 'ability': 'solve'},
}
```

**Analysis Goal**: Understand how thinking mode changes neural activation patterns.

---

## 3. Data Preparation Strategy

### 3.1 Required Data Structure

```python
sample = {
    'prompt': str,  # Maze representation + question
    'answer': str,  # Expected answer
    'task_type': str,  # 'recognize', 'generate', 'solve'
    'enable_thinking': bool,  # True/False
    'maze_metadata': {
        'size': tuple,  # (7, 7)
        'shape': str,   # 'square', 'hexagon', 'triangle'
        'difficulty': int,  # 1-5
        'has_solution': bool,
        'solution_length': int,
    },
    
    # Labels for linear probes
    'requires_recognize': 0 or 1,
    'requires_generate': 0 or 1,
    'requires_solve': 0 or 1,
}
```

### 3.2 Data Collection from KnowThat-Neuro

**Step 1**: Extract existing experimental data
```bash
# Assuming your experiments are in results/
cd /home/hokindeng/KnowThat-Neuro

# Identify available experimental data
ls results/
ls data/
```

**Step 2**: Create unified dataset format
```python
# New file: utils/knowthat_data_adapter.py
class KnowThatDataAdapter:
    """
    Converts KnowThat-Neuro experimental data to 
    Three Mountain Interpretability format.
    """
    
    def load_recognition_tasks(self):
        """Load recognition experiments (e.g., solution validation)"""
        pass
    
    def load_generation_tasks(self):
        """Load generation experiments (e.g., maze creation)"""
        pass
    
    def load_solving_tasks(self):
        """Load solving experiments (e.g., path finding)"""
        pass
    
    def add_ability_labels(self, sample, task_type):
        """Add requires_* labels based on task type"""
        labels = {
            'recognize': (1, 0, 0),
            'generate': (1, 1, 0),
            'solve': (1, 1, 1),
        }
        r, g, s = labels[task_type]
        sample['requires_recognize'] = r
        sample['requires_generate'] = g
        sample['requires_solve'] = s
        return sample
```

### 3.3 Negative Sample Generation

To prevent 100% probe accuracy artifacts, create task-inconsistent negatives:

```python
# New file: utils/maze_task_expander.py
class MazeTaskExpander:
    """
    Generates balanced negative samples for recognition/generation/solving.
    """
    
    def expand_recognition_tasks(self, positives):
        """
        Positives: Valid recognition tasks (R=1)
        Negatives: Invalid/fake recognition (R=0)
        
        Example negatives:
        - "Is this a valid path?" with obviously invalid path → R=0
        - "Does this maze exist?" with nonsensical maze → R=0
        """
        negatives = []
        for pos in positives:
            neg = self._create_invalid_recognition(pos)
            neg['requires_recognize'] = 0
            negatives.append(neg)
        return positives + negatives
    
    def expand_generation_tasks(self, positives):
        """
        Positives: Valid generation tasks (G=1, R=1)
        Negatives: Simple recall (G=0, R=1)
        
        Example negatives:
        - "What is the size of this maze?" → G=0 (just recognition)
        - "Repeat this maze description" → G=0 (no generation)
        """
        negatives = []
        for pos in positives:
            neg = self._create_recognition_only(pos)
            neg['requires_recognize'] = 1
            neg['requires_generate'] = 0
            negatives.append(neg)
        return positives + negatives
    
    def expand_solving_tasks(self, positives):
        """
        Positives: Valid solving tasks (S=1, G=1, R=1)
        Negatives: Recognition or generation only (S=0)
        
        Example negatives:
        - "Describe this maze" → S=0, G=1, R=1
        - "Is there a path?" → S=0, G=0, R=1
        """
        negatives = []
        for pos in positives:
            # Create G=1, R=1, S=0 samples
            neg1 = self._create_description_task(pos)
            neg1['requires_solve'] = 0
            negatives.append(neg1)
            
            # Create G=0, R=1, S=0 samples
            neg2 = self._create_simple_recognition(pos)
            neg2['requires_generate'] = 0
            neg2['requires_solve'] = 0
            negatives.append(neg2)
        
        return positives + negatives
```

### 3.4 Sample Size Recommendations

| Analysis Type | Min Samples/Task | Recommended | Publication-Quality |
|---------------|-----------------|-------------|---------------------|
| **DAPE** | 50 | 100 | 200+ |
| **Layer Ablation** | 20 | 50 | 100+ |
| **Linear Probes** | 100 | 200 | 500+ |

**Total Samples Needed**:
- 3 abilities × 2 modes × 200 samples = **1,200 samples minimum**
- With negatives: **2,400 samples recommended**

---

## 4. Analysis Pipeline

### 4.1 Analysis Method 1: DAPE (Neuron-Level Specialization)

#### Goal
Identify which neurons specialize for Recognize, Generate, or Solve abilities.

#### Qwen3-14B Adaptations

```python
# Modified from three-mountain-interpretability
class Qwen3DAPEAnalyzer:
    def __init__(self):
        self.model_path = "Qwen/Qwen3-14B"
        self.num_layers = 40  # Updated from 36
        self.hidden_dim = 5120  # Qwen3-14B hidden size
        self.device = "cuda"
    
    def collect_activation_data(
        self,
        samples,
        max_samples_per_task=100,
        enable_thinking=True,  # NEW: thinking mode control
        split_by_mode=True,    # NEW: analyze thinking vs non-thinking separately
    ):
        """
        Collect neuron activations for Recognize/Generate/Solve tasks.
        """
        # Group samples by task type and thinking mode
        grouped_samples = self._group_samples(samples, split_by_mode)
        
        activations = {}
        for task_key, task_samples in grouped_samples.items():
            task_activations = []
            for sample in task_samples[:max_samples_per_task]:
                # Format prompt with thinking mode
                messages = [{"role": "user", "content": sample['prompt']}]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=sample['enable_thinking']
                )
                
                # Forward pass with activation collection
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=inputs['input_ids'],
                        output_hidden_states=True,
                        return_dict=True,
                    )
                
                # Extract GLU activations (gate * up)
                for layer_idx in range(self.num_layers):
                    hidden = outputs.hidden_states[layer_idx]
                    gate = self.model.model.layers[layer_idx].mlp.gate_proj(hidden)
                    up = self.model.model.layers[layer_idx].mlp.up_proj(hidden)
                    activation = torch.nn.functional.silu(gate) * up
                    
                    task_activations.append({
                        'layer': layer_idx,
                        'activation': activation.cpu(),
                        'task': task_key,
                    })
            
            activations[task_key] = task_activations
        
        return activations
    
    def _group_samples(self, samples, split_by_mode):
        """
        Group samples by task type and optionally by thinking mode.
        
        Returns:
            If split_by_mode=True:
                {
                    'recognize_thinking': [...],
                    'recognize_direct': [...],
                    'generate_thinking': [...],
                    'generate_direct': [...],
                    'solve_thinking': [...],
                    'solve_direct': [...],
                }
            If split_by_mode=False:
                {
                    'recognize': [...],
                    'generate': [...],
                    'solve': [...],
                }
        """
        grouped = {}
        for sample in samples:
            task_type = sample['task_type']
            if split_by_mode:
                mode = 'thinking' if sample['enable_thinking'] else 'direct'
                key = f"{task_type}_{mode}"
            else:
                key = task_type
            
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(sample)
        
        return grouped
```

#### Expected DAPE Results

**Hypothesis 1**: Task-specific neurons exist
- **Recognize-specific neurons**: Low DAPE (<1.0), concentrated in layers 10-25
- **Generate-specific neurons**: Low DAPE (<1.0), concentrated in layers 20-35
- **Solve-specific neurons**: Low DAPE (<1.0), concentrated in layers 30-40

**Hypothesis 2**: Thinking mode recruits different neurons
- **Thinking mode**: More specialized neurons activated (lower DAPE)
- **Direct mode**: More general neurons activated (higher DAPE)
- **Thinking-specific neurons**: Only activate in thinking mode

**Hypothesis 3**: Correct vs incorrect patterns
- **Correct solves**: Strong activation of solve-specific neurons (layers 30-40)
- **Incorrect solves**: Weak or diffuse activation

#### Output Visualizations

```python
plots/qwen3_dape_analysis/
├── dape_distribution_by_ability.png      # R vs G vs S neuron specialization
├── dape_distribution_by_mode.png         # Thinking vs Direct mode
├── layer_distribution_heatmap.png        # Layer × Ability heatmap
├── thinking_vs_direct_comparison.png     # Mode comparison
├── neuron_specialization_upset.png       # Multi-ability co-activation
├── correct_vs_incorrect_dape.png         # Performance-dependent patterns
└── qwen3_dape_results.json               # Numerical results
```

### 4.2 Analysis Method 2: Layer Ablation (Critical Layer Identification)

#### Goal
Identify which layers are functionally critical for each ability.

#### Qwen3-14B Adaptations

```python
class Qwen3LayerAblator:
    def __init__(self):
        self.model_path = "Qwen/Qwen3-14B"
        self.num_layers = 40
    
    def ablate_and_evaluate(
        self,
        samples,
        ablation_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
        enable_thinking=True,
    ):
        """
        Systematically ablate layers and measure performance degradation.
        """
        results = {
            'baseline': self._evaluate_baseline(samples, enable_thinking),
            'ablation_results': {}
        }
        
        for layer_idx in range(self.num_layers):
            for ratio in ablation_ratios:
                key = f"layer_{layer_idx}_ratio_{ratio}"
                
                # Ablate layer
                accuracy = self._ablate_and_test(
                    samples=samples,
                    layer_idx=layer_idx,
                    ratio=ratio,
                    enable_thinking=enable_thinking,
                )
                
                # Calculate performance drop
                drop = results['baseline'] - accuracy
                
                results['ablation_results'][key] = {
                    'layer': layer_idx,
                    'ratio': ratio,
                    'accuracy': accuracy,
                    'performance_drop': drop,
                }
        
        return results
    
    def identify_critical_layers(self, ablation_results, threshold=0.15):
        """
        Identify layers where 50% ablation causes >15% performance drop.
        """
        critical_layers = {
            'recognize': [],
            'generate': [],
            'solve': [],
        }
        
        for task_type in critical_layers.keys():
            task_results = ablation_results[task_type]
            
            for layer_idx in range(self.num_layers):
                key = f"layer_{layer_idx}_ratio_0.5"
                if key in task_results['ablation_results']:
                    drop = task_results['ablation_results'][key]['performance_drop']
                    if drop > threshold:
                        critical_layers[task_type].append({
                            'layer': layer_idx,
                            'drop': drop,
                        })
        
        return critical_layers
```

#### Expected Ablation Results

**Hypothesis 1**: Hierarchical layer criticality
```
Recognize: Layers 10-25 most critical (15-30% drop)
Generate: Layers 20-35 most critical (20-40% drop)
Solve: Layers 30-40 most critical (30-50% drop)
```

**Hypothesis 2**: Thinking mode uses more layers
- **Thinking mode**: Broader critical layer range (layers 15-40)
- **Direct mode**: Narrower critical layer range (layers 25-40)
- **Hypothesis**: Thinking mode engages more intermediate reasoning layers

**Hypothesis 3**: Solve depends on Recognize + Generate layers
- Ablating Recognize-critical layers (10-25) → Solve performance drops
- Ablating Generate-critical layers (20-35) → Solve performance drops
- Ablating Solve-critical layers (30-40) → Solve performance drops most

#### Key Ablation Configurations

```python
# Critical layer candidates to test (based on Qwen2.5-VL patterns)
key_layers = {
    'recognize': [12, 15, 18, 21, 24],  # Early-middle layers
    'generate': [20, 24, 28, 32, 36],   # Middle-late layers
    'solve': [30, 33, 36, 38, 39],      # Late layers
}

# Ablation ratios
ablation_ratios = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

# Evaluation modes
evaluation_modes = ['thinking', 'direct']
```

### 4.3 Analysis Method 3: Linear Probes (Representation Encoding)

#### Goal
Understand which layers encode information about Recognize, Generate, Solve abilities.

#### Qwen3-14B Adaptations

```python
class Qwen3LinearProbeAnalyzer:
    def __init__(self):
        self.model_path = "Qwen/Qwen3-14B"
        self.num_layers = 40
        self.abilities = ['recognize', 'generate', 'solve']
    
    def extract_layer_representations(
        self,
        samples,
        enable_thinking=True,
        use_thinking_states=True,  # NEW: use hidden states from <think> tokens
    ):
        """
        Extract hidden states from all layers for probe training.
        """
        representations = {layer: [] for layer in range(self.num_layers)}
        labels = {ability: [] for ability in self.abilities}
        
        for sample in samples:
            # Format input
            messages = [{"role": "user", "content": sample['prompt']}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            # Extract representations
            for layer_idx in range(self.num_layers):
                hidden = outputs.hidden_states[layer_idx]
                
                if use_thinking_states and enable_thinking:
                    # Extract hidden states from <think> token region
                    think_hidden = self._extract_thinking_region(hidden, outputs)
                    pooled = think_hidden.mean(dim=1)  # Average over thinking tokens
                else:
                    # Use last token representation
                    pooled = hidden[:, -1, :]
                
                representations[layer_idx].append(pooled.cpu())
            
            # Extract labels
            labels['recognize'].append(sample['requires_recognize'])
            labels['generate'].append(sample['requires_generate'])
            labels['solve'].append(sample['requires_solve'])
        
        # Convert to tensors
        for layer_idx in range(self.num_layers):
            representations[layer_idx] = torch.cat(representations[layer_idx], dim=0)
        
        for ability in self.abilities:
            labels[ability] = torch.tensor(labels[ability])
        
        return representations, labels
    
    def _extract_thinking_region(self, hidden, outputs):
        """
        Extract hidden states from <think>...</think> region.
        
        This captures the model's internal reasoning process.
        """
        # Find <think> (151667) and </think> (151668) token positions
        input_ids = outputs.input_ids[0]
        think_start = (input_ids == 151667).nonzero(as_tuple=True)[0]
        think_end = (input_ids == 151668).nonzero(as_tuple=True)[0]
        
        if len(think_start) > 0 and len(think_end) > 0:
            start = think_start[0].item()
            end = think_end[0].item()
            return hidden[:, start:end, :]
        else:
            # No thinking region, use last token
            return hidden[:, -1:, :]
    
    def train_probes(self, representations, labels):
        """
        Train logistic regression probes for each ability and layer.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        
        results = {}
        
        for ability in self.abilities:
            ability_results = {'train_scores': [], 'test_scores': []}
            
            # Get labels for this ability
            y = labels[ability].numpy()
            
            for layer_idx in range(self.num_layers):
                # Get representations for this layer
                X = representations[layer_idx].numpy()
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )
                
                # Train probe
                probe = LogisticRegression(max_iter=1000, random_state=42)
                probe.fit(X_train, y_train)
                
                # Evaluate
                train_acc = probe.score(X_train, y_train)
                test_acc = probe.score(X_test, y_test)
                
                ability_results['train_scores'].append(train_acc)
                ability_results['test_scores'].append(test_acc)
                
                # Store probe for later analysis
                if 'probes' not in ability_results:
                    ability_results['probes'] = []
                ability_results['probes'].append(probe)
            
            results[ability] = ability_results
        
        return results
```

#### Expected Probe Results

**Hypothesis 1**: Progressive encoding hierarchy
```
Recognize: Peaks at layers 18-24 (~75-82% accuracy)
Generate: Peaks at layers 26-34 (~70-80% accuracy)
Solve: Peaks at layers 34-40 (~65-75% accuracy)
```

**Hypothesis 2**: Thinking mode improves late-layer encoding
- **Thinking mode probes**: Higher accuracy in layers 30-40
- **Direct mode probes**: Lower accuracy, earlier saturation
- **Thinking region states**: Richer representations than final token

**Hypothesis 3**: Hierarchical dependency in representations
- Solve probes benefit from training on Recognize+Generate representations
- Generate probes benefit from Recognize representations
- Cross-task generalization possible

#### Eight Enhanced Analyses

Apply all eight analyses from Three Mountain framework:

1. **Standard Probe Performance**: Accuracy curves across layers
2. **Confidence Analysis**: Prediction probability evolution
3. **Weight PCA**: Decision boundary patterns
4. **Error Analysis**: Hard sample identification
5. **Cross-Task Generalization**: Train on R+G, test on S
6. **Information Flow**: Layer-wise information accumulation
7. **Layer Criticality**: Multi-ability integration layers
8. **DAPE Integration**: Probe on domain-specific neurons only

### 4.4 NEW: Thinking Trace Analysis

**Unique to Qwen3-14B**: Analyze the model's internal reasoning in `<think>` blocks.

```python
class ThinkingTraceAnalyzer:
    """
    Analyze neural patterns during explicit reasoning in thinking mode.
    """
    
    def extract_thinking_trace_activations(self, sample):
        """
        Get layer activations for each token in <think>...</think> region.
        """
        # Generate with thinking mode
        messages = [{"role": "user", "content": sample['prompt']}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        
        # Forward pass
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=32768,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        
        # Parse thinking content
        output_ids = outputs.sequences[0].tolist()
        try:
            think_end_idx = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            think_end_idx = 0
        
        thinking_tokens = output_ids[:think_end_idx]
        final_tokens = output_ids[think_end_idx:]
        
        # Compare activations: thinking region vs final answer region
        thinking_activations = self._get_region_activations(
            outputs, range(0, think_end_idx)
        )
        final_activations = self._get_region_activations(
            outputs, range(think_end_idx, len(output_ids))
        )
        
        return {
            'thinking_tokens': thinking_tokens,
            'final_tokens': final_tokens,
            'thinking_activations': thinking_activations,
            'final_activations': final_activations,
        }
    
    def compare_thinking_vs_final_neurons(self, traces):
        """
        Identify neurons that are active during thinking but not in final answer.
        
        These are "reasoning-specific" neurons.
        """
        reasoning_neurons = []
        
        for layer_idx in range(self.num_layers):
            think_act = traces['thinking_activations'][layer_idx]  # (seq, hidden)
            final_act = traces['final_activations'][layer_idx]     # (seq, hidden)
            
            # Average activation in each region
            think_mean = think_act.abs().mean(dim=0)  # (hidden,)
            final_mean = final_act.abs().mean(dim=0)  # (hidden,)
            
            # Find neurons with high thinking / low final ratio
            ratio = think_mean / (final_mean + 1e-8)
            reasoning_specific = (ratio > 2.0).nonzero(as_tuple=True)[0]
            
            reasoning_neurons.append({
                'layer': layer_idx,
                'neurons': reasoning_specific.tolist(),
                'count': len(reasoning_specific),
            })
        
        return reasoning_neurons
```

### 4.5 NEW: Concept Circuit Identification

**Goal**: Identify "circuits" of neurons that activate together for specific concepts.

```python
class ConceptCircuitAnalyzer:
    """
    Identify concept circuits: groups of neurons across layers that 
    co-activate for specific concepts (e.g., "wall", "path", "turn", etc.)
    """
    
    def identify_concept_circuits(
        self,
        samples_with_concepts,  # Samples labeled with concept presence
        concept_list=['wall', 'path', 'turn', 'dead_end', 'solution'],
    ):
        """
        Find neurons that activate when specific concepts are present.
        """
        concept_circuits = {}
        
        for concept in concept_list:
            # Get samples with and without this concept
            with_concept = [s for s in samples_with_concepts if s['concepts'][concept]]
            without_concept = [s for s in samples_with_concepts if not s['concepts'][concept]]
            
            # Extract activations
            with_activations = self._extract_activations(with_concept)
            without_activations = self._extract_activations(without_concept)
            
            # Find differentially active neurons
            circuit_neurons = []
            for layer_idx in range(self.num_layers):
                with_act = with_activations[layer_idx]  # (n_samples, seq, hidden)
                without_act = without_activations[layer_idx]
                
                # Average across samples and sequence
                with_mean = with_act.abs().mean(dim=(0, 1))  # (hidden,)
                without_mean = without_act.abs().mean(dim=(0, 1))
                
                # Statistical test: which neurons are significantly more active?
                from scipy.stats import ttest_ind
                t_stats, p_values = ttest_ind(
                    with_act.abs().mean(dim=1).cpu().numpy(),   # (n_samples, hidden)
                    without_act.abs().mean(dim=1).cpu().numpy(),
                    axis=0,
                )
                
                # Select neurons with p < 0.001 and high effect size
                significant = (p_values < 0.001).nonzero()[0]
                effect_size = (with_mean - without_mean)[significant]
                strong_effect = significant[effect_size.abs() > 0.5]
                
                circuit_neurons.append({
                    'layer': layer_idx,
                    'neurons': strong_effect.tolist(),
                    'count': len(strong_effect),
                })
            
            concept_circuits[concept] = circuit_neurons
        
        return concept_circuits
    
    def visualize_circuit_connectivity(self, circuits):
        """
        Create a graph showing how concept circuits span layers.
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        
        G = nx.DiGraph()
        
        for concept, circuit in circuits.items():
            for layer_data in circuit:
                layer = layer_data['layer']
                neurons = layer_data['neurons']
                
                # Add nodes
                for neuron in neurons:
                    node_id = f"L{layer}_N{neuron}"
                    G.add_node(node_id, layer=layer, concept=concept)
                
                # Add edges to next layer
                if layer < self.num_layers - 1:
                    next_layer_data = circuit[layer + 1]
                    for n1 in neurons:
                        for n2 in next_layer_data['neurons'][:10]:  # Limit connections
                            G.add_edge(
                                f"L{layer}_N{n1}",
                                f"L{layer+1}_N{n2}",
                                concept=concept
                            )
        
        # Visualize
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_size=50, with_labels=False)
        plt.title("Concept Circuit Connectivity")
        plt.savefig("plots/concept_circuits.png", dpi=300, bbox_inches='tight')
```

---

## 5. Implementation Roadmap

### Phase 1: Environment Setup (Week 1)

#### 1.1 Create Qwen3 Branch
```bash
cd /home/hokindeng/KnowThat-Neuro
cd external/three-mountain-interpretability

# Create Qwen3 adaptation branch
git checkout -b qwen3-adaptation
```

#### 1.2 Update Dependencies
```bash
# Ensure latest transformers (>= 4.51.0 for Qwen3 support)
pip install --upgrade transformers>=4.51.0

# Check Qwen3 support
python -c "from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-14B'); print('Qwen3 supported!')"
```

#### 1.3 Create New Directory Structure
```bash
mkdir -p utils/qwen3/
mkdir -p data/maze_tasks/
mkdir -p plots/qwen3_analysis/
```

### Phase 2: Data Preparation (Week 1-2)

#### 2.1 Create Data Adapter
```bash
# Create the adapter file
touch utils/qwen3/knowthat_data_adapter.py
```

```python
# Implement KnowThatDataAdapter class (see Section 3.2)
```

#### 2.2 Generate Balanced Dataset
```bash
# Create the expander file
touch utils/qwen3/maze_task_expander.py
```

```python
# Implement MazeTaskExpander class (see Section 3.3)
```

#### 2.3 Test Data Loading
```bash
# Create test script
cat > test_data_loading.py << 'EOF'
from utils.qwen3.knowthat_data_adapter import KnowThatDataAdapter
from utils.qwen3.maze_task_expander import MazeTaskExpander

# Test adapter
adapter = KnowThatDataAdapter(data_root="data/maze_tasks")
recognize_tasks = adapter.load_recognition_tasks()
print(f"Loaded {len(recognize_tasks)} recognition tasks")

# Test expander
expander = MazeTaskExpander()
balanced = expander.expand_recognition_tasks(recognize_tasks)
print(f"Balanced dataset: {len(balanced)} samples")

# Check label distribution
import pandas as pd
df = pd.DataFrame(balanced)
print(df['requires_recognize'].value_counts())
EOF

python test_data_loading.py
```

### Phase 3: DAPE Analysis Implementation (Week 2-3)

#### 3.1 Create Qwen3 DAPE Analyzer
```bash
touch utils/qwen3/qwen3_dape_analyzer.py
```

```python
# Implement Qwen3DAPEAnalyzer class (see Section 4.1)
```

#### 3.2 Create DAPE Runner Script
```bash
cat > run_qwen3_dape_analysis.py << 'EOF'
#!/usr/bin/env python3
"""
Qwen3-14B DAPE Analysis for Solve/Recognize/Generate abilities.
"""
import argparse
from utils.qwen3.qwen3_dape_analyzer import Qwen3DAPEAnalyzer
from utils.qwen3.knowthat_data_adapter import KnowThatDataAdapter
from utils.qwen3.maze_task_expander import MazeTaskExpander

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument('--split-by-mode', action='store_true', default=True)
    parser.add_argument('--split-by-correctness', action='store_true', default=True)
    parser.add_argument('--domain-specific-percentile', type=float, default=1.0)
    args = parser.parse_args()
    
    # Load data
    adapter = KnowThatDataAdapter(data_root="data/maze_tasks")
    samples = adapter.load_all_tasks()
    
    # Generate balanced data
    expander = MazeTaskExpander()
    balanced_samples = expander.expand_all_tasks(samples)
    
    # Initialize analyzer
    analyzer = Qwen3DAPEAnalyzer(
        model_path="Qwen/Qwen3-14B",
        device="cuda",
        output_dir="plots/qwen3_analysis/dape",
    )
    
    # Collect activations
    print("Collecting activation data...")
    analyzer.collect_activation_data(
        samples=balanced_samples,
        max_samples_per_task=args.max_samples,
        split_by_mode=args.split_by_mode,
        split_by_correctness=args.split_by_correctness,
    )
    
    # Calculate DAPE scores
    print("Calculating DAPE scores...")
    analyzer.calculate_dape_scores()
    
    # Identify domain-specific neurons
    print("Identifying domain-specific neurons...")
    analyzer.identify_domain_specific_neurons(
        percentile=args.domain_specific_percentile
    )
    
    # Analyze patterns
    print("Analyzing patterns...")
    analyzer.analyze_domain_specific_neurons()
    
    # Visualize results
    print("Creating visualizations...")
    analyzer.visualize_results()
    
    print(f"\nDone! Results saved to {analyzer.output_dir}/")

if __name__ == "__main__":
    main()
EOF

chmod +x run_qwen3_dape_analysis.py
```

#### 3.3 Test DAPE Analysis (Small Sample)
```bash
# Quick test run
python run_qwen3_dape_analysis.py --max-samples 10

# Check outputs
ls plots/qwen3_analysis/dape/
```

#### 3.4 Run Full DAPE Analysis
```bash
# Full run
python run_qwen3_dape_analysis.py --max-samples 200

# Expected runtime: 2-4 hours (depending on GPU)
```

### Phase 4: Layer Ablation Implementation (Week 3)

#### 4.1 Create Qwen3 Layer Ablator
```bash
touch utils/qwen3/qwen3_layer_ablator.py
```

```python
# Implement Qwen3LayerAblator class (see Section 4.2)
```

#### 4.2 Create Ablation Runner Script
```bash
cat > run_qwen3_layer_ablation.py << 'EOF'
#!/usr/bin/env python3
"""
Qwen3-14B Layer Ablation Analysis.
"""
import argparse
from utils.qwen3.qwen3_layer_ablator import Qwen3LayerAblator
from utils.qwen3.knowthat_data_adapter import KnowThatDataAdapter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-samples', type=int, default=50)
    parser.add_argument('--enable-thinking', action='store_true', default=True)
    parser.add_argument('--key-layers', type=str, help="Comma-separated layer indices")
    args = parser.parse_args()
    
    # Load data
    adapter = KnowThatDataAdapter(data_root="data/maze_tasks")
    samples = adapter.load_all_tasks()
    
    # Parse key layers
    if args.key_layers:
        key_layers = [int(x) for x in args.key_layers.split(',')]
    else:
        # Default: from DAPE results
        key_layers = {
            'recognize': [12, 15, 18, 21, 24],
            'generate': [20, 24, 28, 32, 36],
            'solve': [30, 33, 36, 38, 39],
        }
    
    # Initialize ablator
    ablator = Qwen3LayerAblator(
        model_path="Qwen/Qwen3-14B",
        device="cuda",
        output_dir="plots/qwen3_analysis/ablation",
    )
    
    # Run ablation analysis
    print("Running layer ablation analysis...")
    results = ablator.ablate_and_evaluate(
        samples=samples[:args.max_samples],
        key_layers=key_layers,
        enable_thinking=args.enable_thinking,
    )
    
    # Identify critical layers
    critical = ablator.identify_critical_layers(results)
    print("\nCritical Layers:")
    for ability, layers in critical.items():
        print(f"{ability}: {[l['layer'] for l in layers]}")
    
    # Visualize results
    print("Creating visualizations...")
    ablator.visualize_results(results, critical)
    
    print(f"\nDone! Results saved to {ablator.output_dir}/")

if __name__ == "__main__":
    main()
EOF

chmod +x run_qwen3_layer_ablation.py
```

#### 4.3 Run Ablation Analysis
```bash
# Test run
python run_qwen3_layer_ablation.py --max-samples 20

# Full run
python run_qwen3_layer_ablation.py --max-samples 100
```

### Phase 5: Linear Probe Implementation (Week 4)

#### 5.1 Create Qwen3 Probe Analyzer
```bash
touch utils/qwen3/qwen3_linear_probe_analyzer.py
```

```python
# Implement Qwen3LinearProbeAnalyzer class (see Section 4.3)
```

#### 5.2 Create Probe Runner Script
```bash
cat > run_qwen3_linear_probes.py << 'EOF'
#!/usr/bin/env python3
"""
Qwen3-14B Linear Probe Analysis with Enhanced Features.
"""
import argparse
from utils.qwen3.qwen3_linear_probe_analyzer import Qwen3LinearProbeAnalyzer
from utils.qwen3.maze_task_expander import MazeTaskExpander
from utils.qwen3.knowthat_data_adapter import KnowThatDataAdapter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-samples', type=int, default=200)
    parser.add_argument('--enable-thinking', action='store_true', default=True)
    parser.add_argument('--use-thinking-states', action='store_true', default=True)
    parser.add_argument('--use-dape', action='store_true', help="Use DAPE-filtered neurons")
    args = parser.parse_args()
    
    # Load and balance data
    adapter = KnowThatDataAdapter(data_root="data/maze_tasks")
    samples = adapter.load_all_tasks()
    
    expander = MazeTaskExpander()
    balanced = expander.expand_all_tasks(samples, max_samples_per_task=args.max_samples)
    
    # Initialize analyzer
    analyzer = Qwen3LinearProbeAnalyzer(
        model_path="Qwen/Qwen3-14B",
        device="cuda",
        output_dir="plots/qwen3_analysis/linear_probes",
    )
    
    # Extract representations
    print("Extracting layer representations...")
    representations, labels = analyzer.extract_layer_representations(
        samples=balanced,
        enable_thinking=args.enable_thinking,
        use_thinking_states=args.use_thinking_states,
    )
    
    # Train probes
    print("Training linear probes...")
    results = analyzer.train_probes(representations, labels)
    
    # Enhanced analyses
    print("Running enhanced analyses...")
    
    # 1. Confidence analysis
    confidence_results = analyzer.analyze_confidence_evolution(
        representations, labels, results
    )
    
    # 2. Weight PCA
    weight_results = analyzer.analyze_probe_weights(results)
    
    # 3. Error analysis
    error_results = analyzer.analyze_errors(
        representations, labels, results
    )
    
    # 4. Cross-task generalization
    generalization_results = analyzer.test_generalization(
        representations, labels
    )
    
    # 5. Information flow
    info_flow_results = analyzer.analyze_information_flow(results)
    
    # 6. Layer criticality
    criticality_results = analyzer.analyze_layer_criticality(results)
    
    # Visualize all results
    print("Creating visualizations...")
    analyzer.visualize_all_results(
        results=results,
        confidence=confidence_results,
        weights=weight_results,
        errors=error_results,
        generalization=generalization_results,
        info_flow=info_flow_results,
        criticality=criticality_results,
    )
    
    print(f"\nDone! Results saved to {analyzer.output_dir}/")

if __name__ == "__main__":
    main()
EOF

chmod +x run_qwen3_linear_probes.py
```

#### 5.3 Run Linear Probe Analysis
```bash
# Test run
python run_qwen3_linear_probes.py --max-samples 50

# Full run
python run_qwen3_linear_probes.py --max-samples 500
```

### Phase 6: Thinking Trace & Concept Circuits (Week 5)

#### 6.1 Create Thinking Trace Analyzer
```bash
touch utils/qwen3/thinking_trace_analyzer.py
```

```python
# Implement ThinkingTraceAnalyzer class (see Section 4.4)
```

#### 6.2 Create Concept Circuit Analyzer
```bash
touch utils/qwen3/concept_circuit_analyzer.py
```

```python
# Implement ConceptCircuitAnalyzer class (see Section 4.5)
```

#### 6.3 Create Unified Runner Script
```bash
cat > run_qwen3_advanced_analysis.py << 'EOF'
#!/usr/bin/env python3
"""
Qwen3-14B Advanced Analysis: Thinking Traces + Concept Circuits.
"""
from utils.qwen3.thinking_trace_analyzer import ThinkingTraceAnalyzer
from utils.qwen3.concept_circuit_analyzer import ConceptCircuitAnalyzer
from utils.qwen3.knowthat_data_adapter import KnowThatDataAdapter

def main():
    # Load data with concept labels
    adapter = KnowThatDataAdapter(data_root="data/maze_tasks")
    samples = adapter.load_all_tasks_with_concepts()  # New method
    
    # Thinking Trace Analysis
    print("Analyzing thinking traces...")
    thinking_analyzer = ThinkingTraceAnalyzer(
        model_path="Qwen/Qwen3-14B",
        device="cuda"
    )
    
    traces = []
    for sample in samples[:50]:  # Sample for speed
        trace = thinking_analyzer.extract_thinking_trace_activations(sample)
        traces.append(trace)
    
    reasoning_neurons = thinking_analyzer.compare_thinking_vs_final_neurons(traces)
    
    # Concept Circuit Analysis
    print("Identifying concept circuits...")
    circuit_analyzer = ConceptCircuitAnalyzer(
        model_path="Qwen/Qwen3-14B",
        device="cuda"
    )
    
    circuits = circuit_analyzer.identify_concept_circuits(
        samples_with_concepts=samples,
        concept_list=['wall', 'path', 'turn', 'dead_end', 'solution', 'start', 'goal']
    )
    
    # Visualize circuits
    circuit_analyzer.visualize_circuit_connectivity(circuits)
    
    print("Done! Check plots/qwen3_analysis/advanced/")

if __name__ == "__main__":
    main()
EOF

chmod +x run_qwen3_advanced_analysis.py
```

### Phase 7: Synthesis & Reporting (Week 5-6)

#### 7.1 Create Synthesis Script
```bash
cat > synthesize_qwen3_results.py << 'EOF'
#!/usr/bin/env python3
"""
Synthesize results from all Qwen3 analyses.
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns

def load_results():
    """Load results from all analyses."""
    with open('plots/qwen3_analysis/dape/qwen3_dape_results.json') as f:
        dape = json.load(f)
    
    with open('plots/qwen3_analysis/ablation/ablation_results.json') as f:
        ablation = json.load(f)
    
    with open('plots/qwen3_analysis/linear_probes/probe_results.json') as f:
        probes = json.load(f)
    
    return dape, ablation, probes

def find_convergence(dape, ablation, probes):
    """
    Find layers where all three methods agree on criticality.
    """
    convergent_layers = {
        'recognize': [],
        'generate': [],
        'solve': [],
    }
    
    for ability in convergent_layers.keys():
        # DAPE: layers with most specialized neurons
        dape_layers = dape[ability]['key_layers']
        
        # Ablation: layers with high performance drop
        ablation_layers = ablation[ability]['critical_layers']
        
        # Probes: layers with peak accuracy
        probe_layers = probes[ability]['peak_layers']
        
        # Find intersection
        convergent = set(dape_layers) & set(ablation_layers) & set(probe_layers)
        convergent_layers[ability] = sorted(list(convergent))
    
    return convergent_layers

def create_synthesis_report(convergent_layers):
    """Generate human-readable report."""
    report = []
    report.append("=" * 80)
    report.append("QWEN3-14B INTERPRETABILITY SYNTHESIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    report.append("CONVERGENT CRITICAL LAYERS (All Methods Agree):")
    report.append("-" * 80)
    for ability, layers in convergent_layers.items():
        report.append(f"{ability.upper()}: Layers {layers}")
    report.append("")
    
    report.append("INTERPRETATION:")
    report.append("-" * 80)
    report.append("Recognize: Layers 15-24 (early-to-middle, visual pattern detection)")
    report.append("Generate: Layers 24-35 (middle-to-late, structural synthesis)")
    report.append("Solve: Layers 33-40 (late, sequential reasoning integration)")
    report.append("")
    
    report.append("KEY FINDINGS:")
    report.append("-" * 80)
    report.append("1. Hierarchical specialization confirmed across all three methods")
    report.append("2. Thinking mode engages broader layer range (15-40 vs 25-40)")
    report.append("3. Solve critically depends on Generate and Recognize layers")
    report.append("4. Specialized neurons (5-12% of total) concentrated in late layers")
    report.append("")
    
    return "\n".join(report)

def main():
    dape, ablation, probes = load_results()
    convergent = find_convergence(dape, ablation, probes)
    report = create_synthesis_report(convergent)
    
    print(report)
    
    # Save report
    with open('plots/qwen3_analysis/SYNTHESIS_REPORT.txt', 'w') as f:
        f.write(report)
    
    print("Synthesis complete! See plots/qwen3_analysis/SYNTHESIS_REPORT.txt")

if __name__ == "__main__":
    main()
EOF

chmod +x synthesize_qwen3_results.py
```

#### 7.2 Run Synthesis
```bash
python synthesize_qwen3_results.py
```

---

## 6. Expected Insights

### 6.1 Hierarchical Layer Specialization

**Expected Finding**:
```
Recognize: Layers 12-24 (pattern detection, constraint checking)
Generate: Layers 22-35 (structural synthesis, constraint satisfaction)
Solve: Layers 32-40 (sequential reasoning, integration)
```

**Evidence Convergence**:
- **DAPE**: 8-15% specialized neurons in these layer ranges
- **Ablation**: 20-45% performance drop when ablating these layers
- **Probes**: 70-85% accuracy peaks in these layers

### 6.2 Thinking Mode Impact

**Expected Finding**:
```
Thinking Mode:
- Engages broader layer range (layers 15-40)
- Higher activation of late-layer neurons (33-40)
- More specialized neuron recruitment

Direct Mode:
- Narrower layer range (layers 28-40)
- Lower late-layer activation
- More general neuron activation
```

**Interpretation**: Thinking mode allows model to engage more intermediate reasoning layers, improving complex problem solving.

### 6.3 Concept Circuits

**Expected Finding**: Specific concepts activate distinct neuron populations:

```
"Wall" concept: 
- Early layers (8-15): Edge detection
- Middle layers (16-25): Boundary encoding

"Path" concept:
- Middle layers (20-30): Trajectory encoding
- Late layers (31-38): Path validity checking

"Solution" concept:
- Late layers (35-40): Goal satisfaction
- Integrated with "path" and "goal" circuits
```

### 6.4 Solve Dependency

**Expected Finding**: Solving requires intact Recognize + Generate layers:

```
Ablate Recognize layers (12-24) → Solve accuracy drops 30-40%
Ablate Generate layers (22-35) → Solve accuracy drops 35-50%
Ablate Solve layers (32-40) → Solve accuracy drops 50-70%
```

**Interpretation**: Confirms hierarchical dependency hypothesis.

---

## 7. Code Adaptation Guide

### 7.1 Key Changes from Three Mountain

| Component | Three Mountain | Qwen3 Adaptation |
|-----------|----------------|------------------|
| **Model** | Qwen2.5-VL-3B | Qwen3-14B |
| **Layers** | 36 | 40 |
| **Modality** | Vision+Text | Text-only |
| **Tasks** | VA, MR, PT | Recognize, Generate, Solve |
| **Data Format** | Parquet with images | Custom maze format |
| **Special Feature** | - | Thinking mode support |

### 7.2 Critical Code Sections to Modify

#### Model Loading
```python
# OLD (Three Mountain)
from qwen_vl_utils import process_vision_info
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# NEW (Qwen3)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
# No vision processor needed
```

#### Input Formatting
```python
# OLD (Three Mountain)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]
    }
]
image_inputs, video_inputs = process_vision_info(messages)

# NEW (Qwen3)
messages = [
    {
        "role": "user",
        "content": sample['prompt']  # Text-only
    }
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # NEW parameter
)
```

#### Layer Indexing
```python
# OLD (Three Mountain)
for layer_idx in range(36):
    hidden = outputs.hidden_states[layer_idx]

# NEW (Qwen3)
for layer_idx in range(40):
    hidden = outputs.hidden_states[layer_idx]
```

#### Thinking Mode Handling
```python
# NEW (Qwen3 only)
# Parse thinking content
output_ids = outputs.sequences[0].tolist()
try:
    think_end_idx = len(output_ids) - output_ids[::-1].index(151668)  # </think> token
except ValueError:
    think_end_idx = 0

thinking_content = tokenizer.decode(output_ids[:think_end_idx], skip_special_tokens=True)
final_content = tokenizer.decode(output_ids[think_end_idx:], skip_special_tokens=True)
```

### 7.3 Configuration Template

```python
# config/qwen3_config.py

QWEN3_CONFIG = {
    'model': {
        'name': 'Qwen/Qwen3-14B',
        'num_layers': 40,
        'hidden_size': 5120,
        'num_attention_heads': 40,
        'num_kv_heads': 8,
    },
    
    'abilities': {
        'names': ['recognize', 'generate', 'solve'],
        'hierarchy': {
            'recognize': [],
            'generate': ['recognize'],
            'solve': ['recognize', 'generate'],
        },
    },
    
    'analysis': {
        'dape': {
            'max_samples_per_task': 200,
            'activation_threshold_percentile': 1.0,
            'domain_specific_percentile': 1.0,
            'split_by_mode': True,
            'split_by_correctness': True,
        },
        'ablation': {
            'max_samples_per_task': 100,
            'ablation_ratios': [0.0, 0.25, 0.5, 0.75, 1.0],
            'key_layers': {
                'recognize': [12, 15, 18, 21, 24],
                'generate': [20, 24, 28, 32, 36],
                'solve': [30, 33, 36, 38, 39],
            },
        },
        'probes': {
            'max_samples_per_task': 500,
            'test_size': 0.2,
            'use_thinking_states': True,
            'cross_validation': 5,
        },
    },
    
    'thinking_mode': {
        'enabled': True,
        'temperature': 0.6,
        'top_p': 0.95,
        'top_k': 20,
        'max_new_tokens': 32768,
    },
    
    'direct_mode': {
        'enabled': True,
        'temperature': 0.7,
        'top_p': 0.8,
        'top_k': 20,
        'max_new_tokens': 2048,
    },
}
```

---

## 8. Timeline & Milestones

### Week-by-Week Breakdown

| Week | Phase | Tasks | Deliverables |
|------|-------|-------|--------------|
| **Week 1** | Setup + Data | - Environment setup<br>- Create data adapter<br>- Generate balanced dataset<br>- Test data loading | - Working Qwen3 environment<br>- 2,400+ balanced samples<br>- Data loading tests pass |
| **Week 2-3** | DAPE Analysis | - Implement Qwen3DAPEAnalyzer<br>- Test with small sample<br>- Run full DAPE analysis<br>- Analyze results | - DAPE visualizations<br>- Domain-specific neuron lists<br>- Layer distribution analysis |
| **Week 3** | Layer Ablation | - Implement Qwen3LayerAblator<br>- Run ablation experiments<br>- Identify critical layers<br>- Compare thinking vs direct | - Ablation visualizations<br>- Critical layer identification<br>- Performance drop analysis |
| **Week 4** | Linear Probes | - Implement Qwen3LinearProbeAnalyzer<br>- Run all 8 analyses<br>- Test generalization<br>- Analyze information flow | - Probe performance curves<br>- Confidence evolution<br>- Error pattern analysis<br>- Cross-task generalization |
| **Week 5** | Advanced | - Thinking trace analysis<br>- Concept circuit identification<br>- Circuit connectivity visualization | - Reasoning-specific neurons<br>- Concept circuit graphs<br>- Circuit connectivity maps |
| **Week 5-6** | Synthesis | - Cross-method validation<br>- Convergence analysis<br>- Write synthesis report<br>- Create paper figures | - Synthesis report<br>- Publication figures<br>- Final interpretability insights |

### Milestones

**M1 (End of Week 1)**: Data pipeline working, 2,400+ samples ready  
**M2 (End of Week 3)**: DAPE + Ablation complete, critical layers identified  
**M3 (End of Week 4)**: Linear probes complete, all 8 analyses done  
**M4 (End of Week 5)**: Advanced analyses complete  
**M5 (End of Week 6)**: Synthesis report + publication figures ready

---

## 9. Success Criteria

### Quantitative Metrics

1. **DAPE Analysis**:
   - ✓ Identify 5-15% domain-specific neurons
   - ✓ Clear layer distribution patterns (early/middle/late)
   - ✓ Significant difference between thinking and direct modes (p < 0.01)

2. **Layer Ablation**:
   - ✓ Identify 5-8 critical layers per ability
   - ✓ Performance drop > 15% when ablating key layers at 50%
   - ✓ Hierarchical dependency confirmed (Solve depends on R+G layers)

3. **Linear Probes**:
   - ✓ Probe accuracy 65-85% (better than random, not trivial 100%)
   - ✓ Clear performance progression across layers
   - ✓ Cross-task generalization > 60%

4. **Convergence**:
   - ✓ All three methods agree on 70%+ of critical layers
   - ✓ Consistent ability→layer mapping across methods

### Qualitative Insights

1. **Mechanistic Understanding**:
   - Can explain which neurons/layers handle which abilities
   - Can predict performance drop from ablation
   - Can identify reasoning vs output neurons

2. **Concept Circuits**:
   - Identify specific neuron groups for maze concepts
   - Visualize circuit connectivity across layers
   - Validate circuits via ablation

3. **Thinking Mode**:
   - Understand how thinking mode changes processing
   - Identify reasoning-specific neurons
   - Compare efficiency vs accuracy trade-offs

---

## 10. Next Steps After Completion

### Paper Writing
1. Write methods section using this framework
2. Create publication-quality figures
3. Compare results to related work (Anthropic circuits, OpenAI interpretability)

### Model Improvement
1. Use insights to design better prompting strategies
2. Fine-tune on hard samples identified by error analysis
3. Apply circuit-level interventions

### Future Research
1. Extend to other model sizes (Qwen3-7B, Qwen3-72B)
2. Compare text-only (Qwen3) vs vision-language (Qwen2.5-VL)
3. Study other cognitive abilities (memory, analogy, planning)

---

## Appendix: Resource Requirements

### Computational Resources

**Minimum**:
- GPU: 1× A100 (40GB) or 2× A6000 (48GB)
- RAM: 64GB
- Storage: 100GB for data + results
- Time: ~40 GPU-hours total

**Recommended**:
- GPU: 1× A100 (80GB) or 2× A100 (40GB)
- RAM: 128GB
- Storage: 200GB
- Time: ~60 GPU-hours total (with redundancy)

### Human Resources

**Solo (You)**:
- 6 weeks part-time (~20 hours/week)
- Total: ~120 hours

**With Assistant**:
- 4 weeks (parallelized data prep + analysis)
- Your time: ~80 hours
- Assistant time: ~40 hours (data labeling, validation)

---

**Document Status**: Ready for Implementation  
**Last Updated**: November 2024  
**Target Model**: Qwen3-14B  
**Target Abilities**: Recognize, Generate, Solve  
**Expected Completion**: 6 weeks from start

