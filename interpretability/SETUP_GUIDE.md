# Qwen Interpretability Setup Guide

Quick start guide for running interpretability experiments with Qwen models.

## Prerequisites

1. **Python 3.10+** with virtual environment
2. **Qwen API Key** from Alibaba Cloud DashScope
3. **Dependencies** installed

## Quick Setup (5 minutes)

### 1. Activate Environment

```bash
cd /Users/access/KnowWhat/interpretability
source ../venv/bin/activate
```

### 2. Set API Key

```bash
# Add to your .env file or export directly
export DASHSCOPE_API_KEY="your-qwen-api-key-here"

# Verify it's set
echo $DASHSCOPE_API_KEY
```

### 3. Verify Setup

```bash
python check_setup.py
```

## Running Your First Experiment

### Basic Test

```python
from infer.qwen_api import QwenAPI

# Initialize
qwen = QwenAPI(model="qwen-turbo")

# Test
response = qwen.generate("Hello, can you help me solve a maze?")
print(response)
```

### Full Interpretability Experiment

```bash
# Run complete experiment (all tasks, all conditions)
python experiments/qwen_interpretability.py --model qwen-turbo --trials 10

# Quick test (fewer trials)
python experiments/qwen_interpretability.py --model qwen-turbo --trials 2

# Without logit tracking (faster)
python experiments/qwen_interpretability.py --no-logits
```

## Qwen Model Options

| Model | Description | Cost | Best For |
|-------|-------------|------|----------|
| `qwen-turbo` | Fast, cost-effective | Low | Initial testing |
| `qwen-plus` | Balanced | Medium | Standard experiments |
| `qwen-max` | Highest capability | High | Final analysis |
| `qwen-vl` | Vision-language | Medium | Image encoding |

## Understanding the Output

Results are saved to:
```
analysis_results/interpretability/
├── results/
│   └── qwen_turbo_results.json      # Main results
├── activations/
│   └── qwen_turbo_activations.json  # Activation logs
└── logits/
    └── qwen_turbo_logits.json       # Token-level data
```

### Result Format

Each result contains:
```json
{
  "task": "solve|recognize|generate",
  "model": "qwen-turbo",
  "encoding": "matrix|coord_list",
  "size": "5x5|7x7",
  "shape": "square|cross|spiral|triangle|C|Z",
  "trial": 0,
  "is_correct": true,
  "duration": 1.23,
  "logits": {...},
  "response": "..."
}
```

## Expected Results

Based on main experiments, Qwen should show:
- **High solving rate**: 50-90% (procedural knowledge)
- **Low recognition rate**: 5-20% (conceptual knowledge)
- **Low generation rate**: 0-10% (deep conceptual)
- **Dissociation score**: +40% to +80%

## Common Issues

### API Key Not Found
```bash
# Check if set
echo $DASHSCOPE_API_KEY

# Set it
export DASHSCOPE_API_KEY="your-key"
```

### Import Errors
```bash
# Make sure you're in the interpretability directory
cd /Users/access/KnowWhat/interpretability

# Check Python path
python -c "import sys; print(sys.path)"
```

### Rate Limiting
- Qwen has rate limits (QPM/TPM)
- Add delays between requests if needed
- Consider using qwen-turbo for testing

## Next Steps

1. **Run baseline experiment**: `python experiments/qwen_interpretability.py`
2. **Analyze activations**: Check logit patterns in results
3. **Compare with main results**: See `../analysis_results/`
4. **Generate visualizations**: Create attention maps and activation plots

## API Documentation

- **DashScope**: https://help.aliyun.com/zh/dashscope/
- **Qwen Models**: https://qwenlm.github.io/
- **OpenAI Compatible API**: Qwen uses OpenAI-compatible endpoints

## Support

- Main README: `./README.md`
- Main codebase docs: `../README.md`
- Experiment overview: `../experiments/EXPERIMENTS_OVERVIEW.md`

---

**Ready to investigate AI interpretability!** 🔬

