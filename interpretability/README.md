# Interpretability Experiments with Qwen

This directory contains the behavioral experiment framework adapted for interpretability research with Qwen models.

## Overview

This framework enables investigating the internal mechanisms behind the procedural vs. conceptual knowledge dissociation discovered in the main experiments. Using Qwen models, we can analyze:

- **Activation patterns** during maze solving vs. recognition
- **Token-level decision making** for spatial reasoning
- **Internal representations** of maze structures
- **Attention patterns** across different encoding formats

## Key Finding to Investigate

The main experiments revealed:
- **AI Models**: 37-92% maze solving BUT only 1-19% recognition (strong dissociation)
- **Humans**: 99% on all tasks (integrated knowledge)

**Question**: What internal mechanisms cause this dissociation?

## Directory Structure

```
interpretability/
├── core/                    # Maze generation and verification
│   ├── maze_generator.py
│   ├── solution_verifier.py
│   ├── prompt_builder.py
│   └── prompts.py
├── infer/                   # API integrations
│   ├── base_api.py
│   ├── qwen_api.py         # Qwen-specific with interpretability hooks
│   ├── claude_api.py
│   ├── openai_api.py
│   └── ...
├── experiments/             # Experiment runners
│   ├── human_test.py
│   ├── run_control_experiments.py
│   └── qwen_interpretability.py  # New: Qwen-specific experiments
├── scripts/                 # Utility scripts
├── data/                    # Experimental data (symbolic link to main)
├── config.py               # Configuration
├── requirements.txt
└── README.md               # This file
```

## Setup

### 1. Environment Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Set Qwen API key
export DASHSCOPE_API_KEY="your-api-key-here"
```

### 2. Verify Setup

```bash
python check_setup.py
```

## Running Experiments

### Basic Qwen Experiment

```python
from infer.qwen_api import QwenAPI
from core.maze_generator import generate_maze
from core.prompt_builder import encode_maze

# Initialize Qwen
qwen = QwenAPI(model="qwen-turbo")

# Generate and encode maze
maze = generate_maze(size=(5, 5), shape="square")
prompt = encode_maze(maze, encoding="matrix")

# Get response
response = qwen.generate(prompt)
```

### Interpretability Experiment

```python
from infer.qwen_api import QwenAPI

# Initialize with interpretability mode
qwen = QwenAPI(model="qwen-turbo")
qwen.enable_interpretability_mode()

# Run experiment with logit tracking
result = qwen.generate_with_logits(prompt)

# Analyze activations
print(result['logprobs'])  # Token-level decisions
print(qwen.get_activation_logs())  # Full interaction log
```

## Interpretability Features

### 1. **Logit Analysis**
- Token-level probabilities for each decision
- Alternative token distributions
- Confidence patterns across tasks

### 2. **Activation Logging**
- Full prompt-response pairs
- Timestamps for temporal analysis
- Model state tracking

### 3. **Comparative Analysis**
Compare Qwen's internal behavior across:
- Different encodings (matrix vs. coordinate vs. vision)
- Different tasks (solve vs. recognize vs. generate)
- Different maze complexities (5x5 vs. 7x7, simple vs. complex shapes)

## Experimental Design

### Three-Task Paradigm

1. **SOLVE** (Procedural Knowledge)
   - Navigate from start to goal
   - Track: path decisions, uncertainty, correction patterns

2. **RECOGNIZE** (Conceptual Knowledge)
   - Identify maze shape after solving
   - Track: abstraction process, confidence, error patterns

3. **GENERATE** (Deep Conceptual)
   - Create new maze with same shape
   - Track: planning process, constraint satisfaction, shape knowledge

### Encoding Formats

Test all three encodings from main experiments:
- **Matrix**: 2D ASCII grid
- **Coordinate List**: Explicit (row, col) pairs
- **Vision**: Image input (if Qwen vision model available)

## Research Questions

1. **Representation Differences**
   - How does internal representation differ between solving and recognition?
   - Are spatial patterns encoded differently than shape patterns?

2. **Sequential vs. Global Processing**
   - Does the model process mazes sequentially or capture global structure?
   - Where does shape abstraction fail?

3. **Encoding Effects**
   - Why does vision encoding show highest dissociation?
   - What information is lost in each encoding format?

4. **Attention Patterns**
   - What does the model attend to during solving vs. recognition?
   - How do attention patterns correlate with success/failure?

## Data Analysis

Results will be saved to:
```
analysis_results/interpretability/
├── qwen_activations/        # Raw activation data
├── logit_analysis/          # Token-level analysis
├── attention_maps/          # Attention visualizations
└── comparative_analysis/    # Cross-task comparisons
```

## Integration with Main Analysis

The interpretability results can be integrated with the main codebase:

```bash
# Link to main data directory
ln -s ../data ./data
ln -s ../analysis_results ./analysis_results

# Use main analysis pipeline
cd ../analysis
python core/run_full_analysis.py --include-interpretability
```

## Qwen Model Options

Available Qwen models:
- **qwen-turbo**: Fast, cost-effective
- **qwen-plus**: Balanced performance
- **qwen-max**: Highest capability
- **qwen-vl**: Vision-language (for image encoding)

## Configuration

Edit `config.py` to customize:
- Model selection
- Experiment parameters
- Interpretability settings
- Output directories

## Next Steps

1. ✅ Setup complete
2. ⏳ Run baseline Qwen experiments (replicate main findings)
3. ⏳ Collect activation data across all tasks
4. ⏳ Analyze logit patterns and attention
5. ⏳ Compare with other models (Claude, GPT-4o)
6. ⏳ Generate interpretability report

## References

- Main experiment results: `../analysis_results/analysis_report.md`
- Original paper findings: `../paper/`
- Statistical analysis: `../analysis/statistical/`

## Notes

- Always use Pydantic for configuration and data validation
- Avoid try-catch blocks (user preference)
- Save results to `./analysis_results/` directory
- Output both PNG and EPS formats for figures

---

**Status**: Ready for Qwen interpretability experiments
**Last Updated**: 2025-11-22

