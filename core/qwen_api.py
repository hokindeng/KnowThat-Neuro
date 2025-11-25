"""
Qwen API Integration for Maze Experiments
-----------------------------------------
API wrapper for Qwen models with interpretability support.
Includes base API solver class for extensibility.
"""

import os
import time
from pathlib import Path
import re
import numpy as np
from enum import Enum
from typing import Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ConfigDict

from core.prompt_builder import encode_standard_matrix_maze, encode_coordinate_list_maze
from core.maze_generator import *
from core.solution_verifier import get_valid_moves, is_correct_generate, is_correct_recognize
from core.prompts import *
from core.base_api import Status, BaseAPISolver


class QwenConfig(BaseModel):
    """Configuration for Qwen API."""
    model_config = ConfigDict(extra='forbid')
    
    model_name: str = Field(default="qwen-turbo")
    base_url: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    max_tokens: int = Field(default=2000)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.9)


class QwenAPISolver(BaseAPISolver):
    """
    Qwen API integration for maze solving experiments with interpretability hooks.
    
    This class provides:
    - Standard maze-solving capabilities
    - Interpretability experiment support
    - Activation logging for analysis
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-turbo",
        max_tokens: int = 2000,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """
        Initialize Qwen API solver.
        
        Args:
            api_key: API key (defaults to DASHSCOPE_API_KEY env var)
            model: Qwen model name (qwen-turbo, qwen-plus, qwen-max)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        # Initialize parent class
        super().__init__(api_key)
        
        # Interpretability tracking
        self.enable_interpretability = False
        self.activation_logs = []
    
    def _get_api_key(self) -> str:
        """Get API key from environment variables."""
        api_key = os.getenv('DASHSCOPE_API_KEY')
        if not api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY not found in environment variables. "
                "Please set it or pass api_key parameter."
            )
        return api_key
    
    def _initialize_client(self):
        """Initialize the OpenAI-compatible client for Qwen."""
        from openai import OpenAI
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def _make_api_call(self, messages: list, **kwargs) -> str:
        """Make an API call to Qwen and return the response text."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            temperature=kwargs.get('temperature', self.temperature),
            top_p=kwargs.get('top_p', self.top_p)
        )
        return response.choices[0].message.content
    
    def _get_output_directory(self, encoding_type: str) -> str:
        """Get the output directory name for results."""
        return f"{encoding_type}_qwen_results"
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        log_activations: bool = False,
        **kwargs
    ) -> str:
        """
        Generate response from Qwen model.
        
        Args:
            prompt: Input prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            log_activations: Whether to log activations for interpretability
            
        Returns:
            Model response text
        """
        messages = [{"role": "user", "content": prompt}]
        
        call_kwargs = {
            'max_tokens': max_tokens or self.max_tokens,
            'temperature': temperature or self.temperature,
            **kwargs
        }
        
        result = self._make_api_call(messages, **call_kwargs)
        
        # Log for interpretability if requested
        if log_activations and self.enable_interpretability:
            self.activation_logs.append({
                'prompt': prompt,
                'response': result,
                'timestamp': time.time(),
                'model': self.model_name
            })
        
        return result
    
    def generate_with_logits(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate response with logits for interpretability analysis.
        
        Args:
            prompt: Input prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            
        Returns:
            Dictionary with response, logits, and metadata
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        messages = [{"role": "user", "content": prompt}]
        
        # Request logprobs for interpretability
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.top_p,
            logprobs=True,
            top_logprobs=5  # Top 5 alternatives for each token
        )
        
        result = {
            'text': response.choices[0].message.content,
            'logprobs': response.choices[0].logprobs if hasattr(response.choices[0], 'logprobs') else None,
            'model': self.model_name,
            'timestamp': time.time()
        }
        
        # Track token usage
        if hasattr(response, 'usage'):
            result['usage'] = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        
        return result
    
    def enable_interpretability_mode(self):
        """Enable interpretability tracking mode."""
        self.enable_interpretability = True
        self.activation_logs = []
    
    def disable_interpretability_mode(self):
        """Disable interpretability tracking mode."""
        self.enable_interpretability = False
    
    def get_activation_logs(self) -> list:
        """Get logged activations for analysis."""
        return self.activation_logs
    
    def clear_activation_logs(self):
        """Clear activation logs."""
        self.activation_logs = []


def main():
    """Run experiments with Qwen API."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run maze solving experiments with Qwen models')
    parser.add_argument('--model', type=str, default='qwen-turbo',
                       choices=['qwen-turbo', 'qwen-plus', 'qwen-max'],
                       help='Qwen model to use')
    parser.add_argument('--encoding', type=str, nargs='+', default=['matrix', 'coord_list'],
                       choices=['matrix', 'coord_list'],
                       help='Encoding types to use')
    parser.add_argument('--sizes', type=str, nargs='+', default=['5x5'],
                       help='Maze sizes to test')
    
    args = parser.parse_args()
    
    # Parse sizes
    sizes = []
    for size_str in args.sizes:
        parts = size_str.split('x')
        sizes.append((int(parts[0]), int(parts[1])))
    
    # Create solver
    solver = QwenAPISolver(model=args.model)
    
    # Run experiments
    solver.run_all_experiments(
        encoding_types=args.encoding,
        sizes=sizes,
        shapes=None  # Use all shapes
    )


if __name__ == "__main__":
    main()


# Alias for consistency with other API modules
QwenAPI = QwenAPISolver


