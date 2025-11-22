#!/usr/bin/env python3
"""
Run Control Experiments for Direct Shape Generation
---------------------------------------------------
This script executes the control experiments by sending prompts to various
language models and collecting their responses.

Models tested:
- Matrix encoding: Claude Opus 4, Claude 3.5 Sonnet, GPT-4o, Llama 3.1-70B
- Coordinate encoding: Claude Opus 4, Claude 3.5 Sonnet, GPT-4o, Llama 3.1-405B
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import argparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import API classes
from infer.claude_api import ClaudeAPI
from infer.openai_api import OpenAIAPI

# Constants
SHAPES = ["square", "cross", "spiral", "triangle", "C", "Z"]
SIZES = ["5x5", "7x7"]
TRIALS = 10

# Model configurations
MODEL_CONFIGS = {
    "matrix_encoding": {
        "claude4": {
            "api_class": ClaudeAPI,
            "model_name": "claude-opus-4-20250514",
            "results_dir": "matrix_claude4_results"
        },
        "claude": {
            "api_class": ClaudeAPI,
            "model_name": "claude-3-5-sonnet-20241022",
            "results_dir": "matrix_claude_results"
        },
        "openai": {
            "api_class": OpenAIAPI,
            "model_name": "gpt-4o",
            "results_dir": "matrix_openai_results"
        }
    },
    "coordinate_encoding": {
        "claude4": {
            "api_class": ClaudeAPI,
            "model_name": "claude-opus-4-20250514",
            "results_dir": "coord_list_claude4_results"
        },
        "claude": {
            "api_class": ClaudeAPI,
            "model_name": "claude-3-5-sonnet-20241022",
            "results_dir": "coord_list_claude_results"
        },
        "openai": {
            "api_class": OpenAIAPI,
            "model_name": "gpt-4o",
            "results_dir": "coord_list_openai_results"
        }
    }
}


class ControlExperimentRunner:
    """Class to manage and run control experiments."""
    
    def __init__(self, base_dir):
        """
        Initialize the experiment runner.
        
        Args:
            base_dir: Base directory for prompts and results
        """
        self.base_dir = Path(base_dir)
        self.prompts_dir = self.base_dir / "data" / "control_experiment_prompts"
        self.results_dir = self.base_dir / "data" / "control_experiment_results" / "direct_generation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize API instances
        self.apis = {}
        
    def initialize_api(self, api_class, model_name):
        """
        Initialize or get cached API instance.
        
        Args:
            api_class: The API class to instantiate
            model_name: The model name to use
            
        Returns:
            API instance
        """
        key = f"{api_class.__name__}_{model_name}"
        if key not in self.apis:
            # Pass model via keyword for solvers expecting 'model'
            try:
                self.apis[key] = api_class(model=model_name)
            except TypeError:
                # Fallback to positional if signature differs
                self.apis[key] = api_class(model_name)
        return self.apis[key]
    
    def load_prompt(self, encoding, size, shape, trial):
        """
        Load a prompt from file.
        
        Args:
            encoding: 'matrix' or 'coordinate'
            size: '5x5' or '7x7'
            shape: Shape name
            trial: Trial number
            
        Returns:
            str: The prompt content
        """
        prompt_file = self.prompts_dir / f"{encoding}_encoding" / size / shape / f"prompt_trial_{trial}.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
            
        with open(prompt_file, 'r') as f:
            return f.read()
    
    def save_result(self, result, encoding, model_key, size, shape, trial):
        """
        Save experiment result to file.
        
        Args:
            result: The model's response
            encoding: Encoding type
            model_key: Model identifier
            size: Maze size
            shape: Maze shape
            trial: Trial number
        """
        # Get results directory from config
        config = MODEL_CONFIGS[f"{encoding}_encoding"][model_key]
        result_dir = self.results_dir / config["results_dir"] / size / shape
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Save result
        result_file = result_dir / f"trial_{trial}.txt"
        
        # Create result data
        result_data = {
            "timestamp": datetime.now().isoformat(),
            "model": config["model_name"],
            "encoding": encoding,
            "size": size,
            "shape": shape,
            "trial": trial,
            "response": result,
            "prompt_file": str(self.prompts_dir / f"{encoding}_encoding" / size / shape / f"prompt_trial_{trial}.txt")
        }
        
        # Save as text (for compatibility with existing analysis)
        with open(result_file, 'w') as f:
            f.write(f"CONTROL EXPERIMENT - DIRECT GENERATION\n")
            f.write(f"Model: {config['model_name']}\n")
            f.write(f"Encoding: {encoding}\n")
            f.write(f"Size: {size}\n")
            f.write(f"Shape: {shape}\n")
            f.write(f"Trial: {trial}\n")
            f.write(f"Timestamp: {result_data['timestamp']}\n")
            f.write(f"\n{'='*50}\n\n")
            f.write(result)
        
        # Also save as JSON for easier parsing
        json_file = result_file.with_suffix('.json')
        with open(json_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"Saved result to {result_file}")
    
    def run_single_experiment(self, encoding, model_key, size, shape, trial):
        """
        Run a single experiment.
        
        Args:
            encoding: 'matrix' or 'coordinate'
            model_key: Key for model in MODEL_CONFIGS
            size: Maze size
            shape: Maze shape
            trial: Trial number
            
        Returns:
            str: Model response
        """
        # Get model config
        config = MODEL_CONFIGS[f"{encoding}_encoding"][model_key]
        
        # Load prompt
        prompt = self.load_prompt(encoding, size, shape, trial)
        
        # Initialize API
        api = self.initialize_api(config["api_class"], config["model_name"])
        
        # Make API call
        print(f"Calling {config['model_name']} for {encoding} {size} {shape} trial {trial}...")
        
        try:
            response = api.generate(prompt)
            return response
        except Exception as e:
            error_msg = f"Error calling API: {str(e)}"
            print(f"ERROR: {error_msg}")
            return error_msg
    
    def run_all_experiments(self, resume=False, specific_model=None, specific_encoding=None):
        """
        Run all control experiments.
        
        Args:
            resume: If True, skip already completed experiments
            specific_model: If provided, only run this model
            specific_encoding: If provided, only run this encoding
        """
        total_experiments = 0
        completed_experiments = 0
        failed_experiments = 0
        
        # Determine which encodings and models to run
        encodings = [specific_encoding] if specific_encoding else ["matrix", "coordinate"]
        
        for encoding in encodings:
            encoding_key = f"{encoding}_encoding"
            
            # Determine which models to run for this encoding
            if specific_model:
                if specific_model in MODEL_CONFIGS[encoding_key]:
                    models = [specific_model]
                else:
                    print(f"Model {specific_model} not available for {encoding} encoding")
                    continue
            else:
                models = list(MODEL_CONFIGS[encoding_key].keys())
            
            for model_key in models:
                config = MODEL_CONFIGS[encoding_key][model_key]
                print(f"\n{'='*60}")
                print(f"Running experiments for {config['model_name']} with {encoding} encoding")
                print(f"{'='*60}")
                
                for size in SIZES:
                    for shape in SHAPES:
                        for trial in range(TRIALS):
                            total_experiments += 1
                            
                            # Check if already completed (for resume)
                            result_dir = self.results_dir / config["results_dir"] / size / shape
                            result_file = result_dir / f"trial_{trial}.txt"
                            
                            if resume and result_file.exists():
                                print(f"Skipping {encoding} {model_key} {size} {shape} trial {trial} (already completed)")
                                completed_experiments += 1
                                continue
                            
                            # Run experiment
                            try:
                                response = self.run_single_experiment(
                                    encoding, model_key, size, shape, trial
                                )
                                
                                # Save result
                                self.save_result(
                                    response, encoding, model_key, size, shape, trial
                                )
                                
                                completed_experiments += 1
                                
                                # Rate limiting
                                time.sleep(1)  # Adjust based on API rate limits
                                
                            except Exception as e:
                                print(f"Failed to complete experiment: {e}")
                                failed_experiments += 1
                                
                                # Save error
                                self.save_result(
                                    f"EXPERIMENT FAILED: {str(e)}",
                                    encoding, model_key, size, shape, trial
                                )
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Total experiments: {total_experiments}")
        print(f"Completed: {completed_experiments}")
        print(f"Failed: {failed_experiments}")
        print(f"Results saved to: {self.results_dir}")


def main():
    """Main function to run control experiments."""
    parser = argparse.ArgumentParser(description="Run control experiments for direct maze generation")
    parser.add_argument("--resume", action="store_true", help="Resume from where left off")
    parser.add_argument("--model", type=str, help="Run specific model only (claude4, claude, openai)")
    parser.add_argument("--encoding", type=str, help="Run specific encoding only (matrix, coordinate)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model and args.model not in ["claude4", "claude", "openai"]:
        print(f"Invalid model: {args.model}")
        print("Valid models: claude4, claude, openai")
        return
        
    if args.encoding and args.encoding not in ["matrix", "coordinate"]:
        print(f"Invalid encoding: {args.encoding}")
        print("Valid encodings: matrix, coordinate")
        return
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Check if prompts exist
    prompts_dir = project_root / "data" / "control_experiment_prompts"
    if not prompts_dir.exists():
        print(f"Prompts directory not found: {prompts_dir}")
        print("Please run control_prompt_generator.py first")
        return
    
    # Create runner and execute
    runner = ControlExperimentRunner(project_root)
    
    print("Starting control experiments...")
    print(f"Resume: {args.resume}")
    if args.model:
        print(f"Model: {args.model}")
    if args.encoding:
        print(f"Encoding: {args.encoding}")
    
    runner.run_all_experiments(
        resume=args.resume,
        specific_model=args.model,
        specific_encoding=args.encoding
    )


if __name__ == "__main__":
    main()
