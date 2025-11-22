#!/usr/bin/env python3
"""
Qwen Interpretability Experiments
---------------------------------
Run behavioral experiments with Qwen models to investigate internal mechanisms
behind procedural vs. conceptual knowledge dissociation.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import time
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from infer.qwen_api import QwenAPI
from core.maze_generator import *
from core.prompt_builder import *
from core.solution_verifier import *


class ExperimentConfig(BaseModel):
    """Configuration for interpretability experiments."""
    model_config = ConfigDict(extra='forbid')
    
    model_name: str = Field(default="qwen-turbo")
    maze_sizes: List[tuple] = Field(default=[(5, 5), (7, 7)])
    shapes: List[str] = Field(default=["square", "cross", "spiral", "triangle", "C", "Z"])
    encodings: List[str] = Field(default=["matrix", "coord_list"])
    trials_per_condition: int = Field(default=10)
    enable_logit_tracking: bool = Field(default=True)
    save_activations: bool = Field(default=True)
    output_dir: Path = Field(default=Path("./analysis_results/interpretability"))


class QwenInterpretabilityExperiment:
    """Run interpretability experiments with Qwen models."""
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration (uses defaults if None)
        """
        self.config = config or ExperimentConfig()
        self.qwen = QwenAPI(model=self.config.model_name)
        
        # Enable interpretability mode
        if self.config.save_activations:
            self.qwen.enable_interpretability_mode()
        
        # Create output directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / "activations").mkdir(exist_ok=True)
        (self.config.output_dir / "logits").mkdir(exist_ok=True)
        (self.config.output_dir / "results").mkdir(exist_ok=True)
        
        self.results = []
    
    def run_solve_task(
        self,
        maze: np.ndarray,
        encoding: str,
        size: tuple,
        shape: str,
        trial: int
    ) -> Dict[str, Any]:
        """
        Run maze solving task with interpretability tracking.
        
        Args:
            maze: Maze array
            encoding: Encoding type ('matrix' or 'coord_list')
            size: Maze size tuple
            shape: Maze shape name
            trial: Trial number
            
        Returns:
            Result dictionary with response and interpretability data
        """
        # Build prompt
        if encoding == "matrix":
            prompt = encode_standard_matrix_maze(maze) + "\nYour task is to navigate from P to G. Provide your path as a sequence of moves."
        else:
            prompt = encode_coordinate_list_maze(maze) + "\nYour task is to navigate from P to G. Provide your path as coordinates."
        
        start_time = time.time()
        
        # Get response with logits if enabled
        if self.config.enable_logit_tracking:
            result = self.qwen.generate_with_logits(prompt)
            response = result['text']
            logits = result.get('logprobs', None)
        else:
            response = self.qwen.generate(prompt)
            logits = None
        
        duration = time.time() - start_time
        
        # Verify solution
        is_correct = verify_maze_solution(maze, response)
        
        result_data = {
            'task': 'solve',
            'model': self.config.model_name,
            'encoding': encoding,
            'size': f"{size[0]}x{size[1]}",
            'shape': shape,
            'trial': trial,
            'prompt': prompt,
            'response': response,
            'is_correct': is_correct,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'logits': logits
        }
        
        return result_data
    
    def run_recognize_task(
        self,
        maze: np.ndarray,
        encoding: str,
        size: tuple,
        shape: str,
        trial: int
    ) -> Dict[str, Any]:
        """
        Run shape recognition task with interpretability tracking.
        
        Args:
            maze: Maze array
            encoding: Encoding type
            size: Maze size tuple
            shape: True shape name
            trial: Trial number
            
        Returns:
            Result dictionary with response and interpretability data
        """
        # Build prompt
        if encoding == "matrix":
            prompt = encode_standard_matrix_maze(maze) + "\nWhat shape do the empty spaces (0, P, G) form? Choose from: square, cross, spiral, triangle, C, or Z."
        else:
            prompt = encode_coordinate_list_maze(maze) + "\nWhat shape do the empty spaces form? Choose from: square, cross, spiral, triangle, C, or Z."
        
        start_time = time.time()
        
        # Get response with logits
        if self.config.enable_logit_tracking:
            result = self.qwen.generate_with_logits(prompt)
            response = result['text']
            logits = result.get('logprobs', None)
        else:
            response = self.qwen.generate(prompt)
            logits = None
        
        duration = time.time() - start_time
        
        # Check if correct
        is_correct = is_correct_recognize(response, shape)
        
        result_data = {
            'task': 'recognize',
            'model': self.config.model_name,
            'encoding': encoding,
            'size': f"{size[0]}x{size[1]}",
            'shape': shape,
            'trial': trial,
            'prompt': prompt,
            'response': response,
            'is_correct': is_correct,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'logits': logits
        }
        
        return result_data
    
    def run_generate_task(
        self,
        encoding: str,
        size: tuple,
        shape: str,
        trial: int
    ) -> Dict[str, Any]:
        """
        Run maze generation task with interpretability tracking.
        
        Args:
            encoding: Encoding type
            size: Maze size tuple
            shape: Target shape name
            trial: Trial number
            
        Returns:
            Result dictionary with response and interpretability data
        """
        # Build prompt
        if encoding == "matrix":
            prompt = f"Generate a {size[0]}x{size[1]} maze where the empty spaces form a {shape} shape. Use 1 for walls, 0 for empty spaces, P for player, and G for goal."
        else:
            prompt = f"Generate a {size[0]}x{size[1]} maze where the empty spaces form a {shape} shape. Provide wall coordinates, empty coordinates, player position, and goal position."
        
        start_time = time.time()
        
        # Get response with logits
        if self.config.enable_logit_tracking:
            result = self.qwen.generate_with_logits(prompt)
            response = result['text']
            logits = result.get('logprobs', None)
        else:
            response = self.qwen.generate(prompt)
            logits = None
        
        duration = time.time() - start_time
        
        # Verify generated maze
        is_correct = verify_generated_maze(response, shape, encoding)
        
        result_data = {
            'task': 'generate',
            'model': self.config.model_name,
            'encoding': encoding,
            'size': f"{size[0]}x{size[1]}",
            'shape': shape,
            'trial': trial,
            'prompt': prompt,
            'response': response,
            'is_correct': is_correct,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'logits': logits
        }
        
        return result_data
    
    def run_full_experiment(self):
        """Run complete experiment across all conditions."""
        print(f"Starting Qwen Interpretability Experiment")
        print(f"Model: {self.config.model_name}")
        print(f"Conditions: {len(self.config.maze_sizes)} sizes × {len(self.config.shapes)} shapes × {len(self.config.encodings)} encodings × {self.config.trials_per_condition} trials")
        print("="*60)
        
        total_experiments = len(self.config.maze_sizes) * len(self.config.shapes) * len(self.config.encodings) * self.config.trials_per_condition * 3  # 3 tasks
        completed = 0
        
        for size in self.config.maze_sizes:
            for shape in self.config.shapes:
                for encoding in self.config.encodings:
                    for trial in range(self.config.trials_per_condition):
                        # Generate maze
                        maze = generate_maze_with_shape(size, shape)
                        
                        # Run all three tasks
                        print(f"\nRunning: {size} {shape} {encoding} trial {trial}")
                        
                        # Task 1: Solve
                        solve_result = self.run_solve_task(maze, encoding, size, shape, trial)
                        self.results.append(solve_result)
                        completed += 1
                        print(f"  Solve: {'✓' if solve_result['is_correct'] else '✗'} ({completed}/{total_experiments})")
                        
                        # Task 2: Recognize
                        recognize_result = self.run_recognize_task(maze, encoding, size, shape, trial)
                        self.results.append(recognize_result)
                        completed += 1
                        print(f"  Recognize: {'✓' if recognize_result['is_correct'] else '✗'} ({completed}/{total_experiments})")
                        
                        # Task 3: Generate
                        generate_result = self.run_generate_task(encoding, size, shape, trial)
                        self.results.append(generate_result)
                        completed += 1
                        print(f"  Generate: {'✓' if generate_result['is_correct'] else '✗'} ({completed}/{total_experiments})")
                        
                        # Save intermediate results
                        if completed % 10 == 0:
                            self.save_results()
        
        # Final save
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save experiment results to disk."""
        output_file = self.config.output_dir / "results" / f"qwen_{self.config.model_name}_results.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"  💾 Saved results to {output_file}")
        
        # Save activation logs if enabled
        if self.config.save_activations:
            activation_file = self.config.output_dir / "activations" / f"qwen_{self.config.model_name}_activations.json"
            activations = self.qwen.get_activation_logs()
            
            with open(activation_file, 'w') as f:
                json.dump(activations, f, indent=2)
    
    def print_summary(self):
        """Print experiment summary."""
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        
        # Calculate success rates by task
        tasks = {}
        for result in self.results:
            task = result['task']
            if task not in tasks:
                tasks[task] = {'correct': 0, 'total': 0}
            tasks[task]['total'] += 1
            if result['is_correct']:
                tasks[task]['correct'] += 1
        
        for task, stats in tasks.items():
            rate = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"{task.capitalize()}: {stats['correct']}/{stats['total']} ({rate:.1f}%)")
        
        # Calculate dissociation score
        solve_rate = (tasks['solve']['correct'] / tasks['solve']['total']) * 100
        recognize_rate = (tasks['recognize']['correct'] / tasks['recognize']['total']) * 100
        generate_rate = (tasks['generate']['correct'] / tasks['generate']['total']) * 100
        dissociation = solve_rate - ((recognize_rate + generate_rate) / 2)
        
        print(f"\nDissociation Score: {dissociation:.1f}%")
        print(f"  (Solve - (Recognize + Generate) / 2)")
        
        print(f"\nTotal experiments: {len(self.results)}")
        print(f"Results saved to: {self.config.output_dir}")


def main():
    """Main function to run Qwen interpretability experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Qwen interpretability experiments")
    parser.add_argument("--model", type=str, default="qwen-turbo", help="Qwen model name")
    parser.add_argument("--trials", type=int, default=10, help="Trials per condition")
    parser.add_argument("--no-logits", action="store_true", help="Disable logit tracking")
    
    args = parser.parse_args()
    
    # Create config
    config = ExperimentConfig(
        model_name=args.model,
        trials_per_condition=args.trials,
        enable_logit_tracking=not args.no_logits
    )
    
    # Run experiment
    experiment = QwenInterpretabilityExperiment(config)
    experiment.run_full_experiment()


if __name__ == "__main__":
    main()

