# Behavioral Experiments: LLM Spatial Reasoning in Maze Navigation

## Overview

This document describes the behavioral experiments designed to evaluate Large Language Models' (LLMs) spatial reasoning, pattern recognition, and generative capabilities using structured maze navigation tasks.

## Experimental Design

### Three-Phase Task Structure

Each experimental trial consists of three sequential phases:

#### **Phase 1: Interactive Maze Solving**
The model must navigate from a player position (P) to a goal position (G) through a 2D maze.

**Task Characteristics:**
- **Iterative Navigation**: Model provides coordinates for next move
- **State Updates**: Maze state updates after each valid move  
- **Maximum Iterations**: 16 moves maximum per maze
- **Success Condition**: Reaching the goal position (G)
- **Failure Conditions**: Invalid move or exceeding move limit

**Movement Rules:**
- **4-directional movement** (up, down, left, right): square, C, spiral shapes
- **8-directional movement** (includes diagonals): cross, triangle, Z shapes

#### **Phase 2: Shape Recognition**
After maze completion (success or failure), the model must identify the geometric pattern formed by the maze corridors.

**Task Requirements:**
- Analyze all empty spaces (not just the solution path)
- Identify the overall geometric shape
- Provide shape name with reasoning

**Target Shapes:**
- Square (and variants: box, cube, quadrilateral)
- Cross (and variants: X-shape, crossed lines)
- Spiral (and variants: helix, coil, whorl)
- Triangle (and variants: pyramid, trilateral)
- C-shape (and variants: crescent, semi-circle)
- Z-shape (and variants: zigzag, lightning bolt)

#### **Phase 3: Shape Generation**
The model must create a new maze with the same shape pattern as the original.

**Generation Requirements:**
- Same dimensions as original (5×5)
- Same geometric shape formed by corridors
- Valid path from start (P) to goal (G)
- NOT an identical copy of the original
- Properly encoded using the specified format

---

## Independent Variables

### 1. Maze Size
- **5×5 grids** (only size used in current experiments)
- Rationale: Balances complexity with computational feasibility

### 2. Geometric Shapes (6 types)
1. **Square**: Rectangular/quadrilateral path patterns
2. **Cross**: X-shaped or intersecting diagonal paths
3. **Spiral**: Clockwise or counterclockwise curved patterns
4. **Triangle**: Triangular path configurations (various orientations)
5. **C**: C-shaped or crescent patterns (4 orientations)
6. **Z**: Z-shaped zigzag patterns

### 3. Encoding Formats (2 types)

#### Matrix Encoding
Visual grid representation using characters:
```
1 0 0 P
0 1 1 0
0 0 0 0
1 1 0 G
```
- `1` = Wall
- `0` = Empty space/path
- `P` = Player position
- `G` = Goal position

#### Coordinate List Encoding
Explicit coordinate listing:
```
Walls: (0,0), (1,1), (1,2), (3,0), (3,1)
Empty: (0,1), (0,2), (1,0), (2,0), (2,1), (2,2), (2,3), (3,2)
Player position: (0,3)
Goal: (3,3)
```

### 4. Model Variants
- **Qwen-turbo**: Fastest, baseline performance
- **Qwen-plus**: Enhanced capabilities
- **Qwen-max**: Maximum performance model

---

## Sample Sizes

**Per Experimental Condition:**
- **5×5 mazes**: 30 samples per shape
- **Total mazes per encoding**: 6 shapes × 30 samples = **180 mazes**
- **Total experimental trials**: 180 mazes × 2 encodings = **360 trials per model**

---

## Dependent Variables (Evaluation Metrics)

### Phase 1: Solving Performance
1. **Success Rate**: Percentage of mazes solved successfully
2. **Number of Moves**: Steps taken to reach goal (efficiency metric)
3. **Failure Type**: Invalid move vs. timeout/max iterations
4. **Move Validity Rate**: Proportion of valid moves made

### Phase 2: Recognition Accuracy
1. **Correct Recognition Rate**: Binary success/failure per trial
2. **Shape Confusion Matrix**: Which shapes are confused with others
3. **Recognition by Encoding**: Performance differences between formats

### Phase 3: Generation Accuracy
1. **Valid Generation Rate**: Percentage of properly formatted mazes
2. **Shape Preservation**: Correctness of generated shape
3. **Novelty**: Non-duplication of original maze
4. **Path Validity**: Generated maze has valid solution path

---

## Experimental Hypotheses

### H1: Encoding Effect
**Hypothesis**: Matrix encoding will outperform coordinate list encoding due to spatial visualization affordances.

**Rationale**: Visual matrix format may better leverage pre-training on code/grid structures.

### H2: Shape Complexity
**Hypothesis**: Recognition and generation accuracy will vary by shape complexity.

**Predicted Difficulty Ranking** (easy → hard):
1. Square (most regular)
2. Cross (symmetric)
3. C-shape (simple asymmetry)
4. Triangle (moderate complexity)
5. Z-shape (diagonal complexity)
6. Spiral (highest complexity)

### H3: Model Capability
**Hypothesis**: Performance will scale with model size (turbo < plus < max).

### H4: Task Phase Difficulty
**Hypothesis**: Solving > Recognition > Generation (decreasing performance).

**Rationale**: 
- Solving: Guided, step-by-step with feedback
- Recognition: Single-shot classification
- Generation: Requires abstract understanding and creative synthesis

---

## Data Collection

### Output Structure
Each trial generates a text file containing:
```
[Initial Prompt + Maze]
[Model Response - Move 1]
[Updated Maze]
[Model Response - Move 2]
...
[Recognition Prompt]
[Model Recognition Response]
[Generation Prompt]
[Model Generation Response]

SOLVE: SUCCESS/FAIL
RECOGNIZE: SUCCESS/FAIL
GENERATE: SUCCESS/FAIL
```

### File Organization
```
{encoding}_qwen_results/
├── 5x5/
│   ├── square/
│   │   ├── 5x5_square_0.npy/
│   │   │   ├── 0.txt
│   │   │   ├── 1.txt
│   │   │   └── ... (30 trials)
│   │   └── ...
│   ├── cross/
│   ├── spiral/
│   ├── triangle/
│   ├── C/
│   └── Z/
```

---

## Analysis Plan

### Primary Analyses
1. **Overall Performance by Model**: Aggregate success rates across all phases
2. **Encoding Comparison**: Matrix vs. Coordinate performance
3. **Shape Effects**: ANOVA on shape difficulty
4. **Phase Progression**: Success rate degradation across phases

### Secondary Analyses
1. **Move Efficiency**: Average moves per successful solve by shape
2. **Error Analysis**: Categorization of failure modes
3. **Correlation Analysis**: Solve success → Recognition → Generation
4. **Interaction Effects**: Model × Encoding × Shape

### Interpretability Extensions
The codebase includes hooks for:
- **Activation logging**: Track internal model states during solving
- **Logit analysis**: Examine token-level decision making
- **Attention patterns**: (future work) visualize attention to maze regions

---

## Scientific Questions

### Core Questions
1. **Do LLMs possess genuine spatial reasoning capabilities?**
   - Or are they pattern matching on training data?

2. **How do representation formats affect spatial cognition?**
   - Does visual/matrix format provide advantages?

3. **Can LLMs form abstract geometric concepts?**
   - Recognition and generation test abstraction ability

4. **What are the limits of LLM spatial reasoning?**
   - At what complexity do they fail?

### Implications
- **AI Safety**: Understanding spatial reasoning for robotics applications
- **Model Design**: Insights for improving spatial reasoning architectures
- **Cognitive Science**: Comparisons with human spatial cognition
- **Benchmark Development**: Establishing standardized spatial reasoning tests

---

## Execution

### Running Experiments

**Command Line Interface:**
```bash
# Run all experiments with default settings (qwen-turbo, 5x5, all encodings)
python -m core.qwen_api

# Specify model
python -m core.qwen_api --model qwen-plus

# Specify encodings
python -m core.qwen_api --encoding matrix

# Specify sizes (only 5x5 supported)
python -m core.qwen_api --sizes 5x5
```

**Programmatic Usage:**
```python
from core import QwenAPISolver

# Initialize solver
solver = QwenAPISolver(model='qwen-turbo')

# Run all experiments
solver.run_all_experiments(
    encoding_types=['matrix', 'coord_list'],
    sizes=[(5, 5)],
    shapes=['square', 'cross', 'spiral', 'triangle', 'C', 'Z']
)
```

### Environment Setup
```bash
# Required environment variable
export DASHSCOPE_API_KEY='your-api-key-here'

# Install dependencies
pip install -r requirements.txt
```

---

## Limitations and Future Directions

### Current Limitations
1. **Single size**: Only 5×5 mazes tested
2. **Limited shapes**: 6 geometric patterns
3. **No vision models**: Text-only encodings
4. **Static mazes**: No dynamic/adversarial generation

### Future Extensions
1. **Size scaling**: Test 7×7, 9×9, 11×11 mazes
2. **Shape expansion**: Include U, N, and complex composite shapes
3. **Vision experiments**: Test multimodal models with image inputs
4. **Dynamic difficulty**: Adaptive maze generation based on performance
5. **Human comparison**: Benchmark against human spatial reasoning
6. **Cross-model analysis**: Test GPT, Claude, Llama variants

---

## References

**Related Work:**
- Spatial reasoning in neural networks
- Compositional generalization in LLMs
- Pattern recognition and abstraction
- Maze solving as cognitive benchmark

**Code Repository Structure:**
- `core/maze_generator.py`: Maze creation algorithms
- `core/prompt_builder.py`: Encoding format implementations  
- `core/solution_verifier.py`: Evaluation logic
- `core/base_api.py`: Base experiment framework
- `core/qwen_api.py`: Qwen-specific implementation

---

## Appendix: Evaluation Functions

### Recognition Validation
Checks if model response contains correct shape keywords (case-insensitive):
- Square: `['square', 'box', 'cube', 'symmetrical rectangle', ...]`
- Triangle: `['triangle', 'pyramid', 'trilateral', 'isosceles', ...]`
- Spiral: `['spiral', 'helix', 'coil', 'whorl', 'swirl', ...]`
- Cross: `['cross', 'times', 'multiplication', 'X', ...]`
- C: `['C', 'crescent', 'half-circle', 'semi-circle', ...]`
- Z: `['Z', 'zigzag', 'lightning bolt', ...]`

### Generation Validation
Verifies generated maze:
1. Matches one of the valid shape templates for that size
2. Is not identical to the original maze
3. Contains proper encoding (walls, paths, start, goal)
4. Has at least one valid solution path

### Solution Validation
Traces coordinate sequence backwards from goal to start:
1. Find last occurrence of goal position in response
2. Walk backwards through coordinates
3. Verify each move is valid (adjacent cell, not wall)
4. Success if reach start position

---

**Document Version**: 1.0  
**Last Updated**: November 24, 2025  
**Experiment Status**: Ready to execute

