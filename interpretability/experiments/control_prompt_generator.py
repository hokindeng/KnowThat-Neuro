#!/usr/bin/env python3
"""
Control Prompt Generator for Direct Shape Generation Experiment
--------------------------------------------------------------
This script generates control prompts for testing language models' ability
to directly generate mazes of specified shapes without prior solving/recognition.

The prompts include:
- The first half of the solve prompt (encoding information)
- Direct instruction to generate a maze of a specific shape and size
"""

import os
from pathlib import Path

# Constants
SHAPES = ["square", "cross", "spiral", "triangle", "C", "Z"]
SIZES = ["5x5", "7x7"]
TRIALS = 10

# Matrix encoding instruction (from INITIAL_SOLVE_PROMPT_1)
MATRIX_ENCODING_INSTRUCTION = """You are tasked with generating a maze. The maze is a 2D grid with walls, empty spaces, a player position, and a goal position. The maze has the following encoding:
* Walls are represented by a '1'
* Empty spaces are represented by a '0'
* The player position is represented by 'P'
* The goal position is represented by 'G'

"""

# Coordinate encoding instruction (from INITIAL_SOLVE_PROMPT_COORD_1)
COORDINATE_ENCODING_INSTRUCTION = """You are tasked with generating a maze. The maze is a 2D grid encoded using coordinates. For reference, the coordinates of the top left corner of the maze are (0,0).

"""

# Generation instruction template
MATRIX_GENERATION_TEMPLATE = """Your task is to generate a {size} maze where the empty spaces (including P and G positions) form a {shape} shape.

Requirements:
1. The maze must be exactly {size} in dimensions
2. The pattern of non-wall cells (0, P, G) must clearly form a {shape}
3. There must be a valid path from P to G
4. P and G should be placed at different positions within the shape
5. Use spaces between characters in each row (e.g., "1 0 1 0 1")

Please generate the maze:
"""

COORDINATE_GENERATION_TEMPLATE = """Your task is to generate a {size} maze where the empty spaces form a {shape} shape.

Requirements:
1. List all wall coordinates as "Walls: (row, col), (row, col), ..."
2. List all empty space coordinates as "Empty: (row, col), (row, col), ..."
3. Specify "Player position: (row, col)"
4. Specify "Goal: (row, col)"
5. The pattern of empty spaces (including player and goal positions) must clearly form a {shape}
6. There must be a valid path from Player position to Goal
7. Ensure all coordinates are within the {size} grid (0 to {max_coord} for both row and column)

Please generate the maze:
"""


def create_control_prompt(encoding_type, shape, size):
    """
    Create a control prompt for direct shape generation.
    
    Args:
        encoding_type: 'matrix' or 'coordinate'
        shape: One of the SHAPES
        size: '5x5' or '7x7'
    
    Returns:
        str: The complete prompt
    """
    if encoding_type == "matrix":
        base_instruction = MATRIX_ENCODING_INSTRUCTION
        generation_template = MATRIX_GENERATION_TEMPLATE
    elif encoding_type == "coordinate":
        base_instruction = COORDINATE_ENCODING_INSTRUCTION
        generation_template = COORDINATE_GENERATION_TEMPLATE
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    # Calculate max coordinate for coordinate encoding
    size_int = int(size.split('x')[0])
    max_coord = size_int - 1
    
    # Format the generation instruction
    generation_instruction = generation_template.format(
        size=size,
        shape=shape,
        max_coord=max_coord
    )
    
    return base_instruction + generation_instruction


def generate_all_prompts(output_dir):
    """
    Generate all control prompts and save them to files.
    
    Args:
        output_dir: Base directory for saving prompts
    """
    output_path = Path(output_dir)
    
    # Create prompts for each encoding type
    for encoding in ["matrix", "coordinate"]:
        encoding_dir = output_path / f"{encoding}_encoding"
        encoding_dir.mkdir(parents=True, exist_ok=True)
        
        # Create prompts for each size
        for size in SIZES:
            size_dir = encoding_dir / size
            size_dir.mkdir(exist_ok=True)
            
            # Create prompts for each shape
            for shape in SHAPES:
                shape_dir = size_dir / shape
                shape_dir.mkdir(exist_ok=True)
                
                # Create the prompt
                prompt = create_control_prompt(encoding, shape, size)
                
                # Save multiple copies for trials
                for trial in range(TRIALS):
                    prompt_file = shape_dir / f"prompt_trial_{trial}.txt"
                    with open(prompt_file, 'w') as f:
                        f.write(prompt)
                
                print(f"Generated prompts for {encoding} {size} {shape}")
    
    # Also save a single example of each unique prompt for reference
    examples_dir = output_path / "prompt_examples"
    examples_dir.mkdir(exist_ok=True)
    
    for encoding in ["matrix", "coordinate"]:
        for size in SIZES:
            for shape in SHAPES:
                prompt = create_control_prompt(encoding, shape, size)
                example_file = examples_dir / f"{encoding}_{size}_{shape}.txt"
                with open(example_file, 'w') as f:
                    f.write(prompt)
    
    print(f"\nAll prompts generated successfully in {output_path}")
    print(f"Total unique prompts: {len(['matrix', 'coordinate']) * len(SIZES) * len(SHAPES)}")
    print(f"Total prompt files (with trials): {len(['matrix', 'coordinate']) * len(SIZES) * len(SHAPES) * TRIALS}")


def main():
    """Main function to generate all control prompts."""
    # Define output directory
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "control_experiment_prompts"
    
    # Generate all prompts
    generate_all_prompts(output_dir)
    
    # Print summary
    print("\nPrompt generation complete!")
    print(f"Prompts saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Run experiments with run_control_experiments.py")
    print("2. Verify results with verify_control_generations.py")
    print("3. Analyze results with analyze_control_results.py")


if __name__ == "__main__":
    main()
