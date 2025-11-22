"""
Qwen API Integration for Maze Experiments
-----------------------------------------
API wrapper for Qwen models with interpretability support.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import os
import time
from pydantic import BaseModel, Field, ConfigDict

from .base_api import BaseAPISolver


class QwenConfig(BaseModel):
    """Configuration for Qwen API."""
    model_config = ConfigDict(extra='forbid')
    
    model_name: str = Field(default="qwen-turbo")
    api_key: Optional[str] = Field(default=None)
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
        model: str = "qwen-turbo",
        api_key: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Initialize Qwen API solver.
        
        Args:
            model: Qwen model name (qwen-turbo, qwen-plus, qwen-max)
            api_key: API key (defaults to DASHSCOPE_API_KEY env var)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        """
        super().__init__(model=model)
        
        self.config = QwenConfig(
            model_name=model,
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        if not self.config.api_key:
            raise ValueError(
                "Qwen API key not found. Set DASHSCOPE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Import OpenAI client (Qwen compatible API)
        import openai
        self.client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
        
        # Interpretability tracking
        self.enable_interpretability = False
        self.activation_logs = []
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        log_activations: bool = False
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
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        
        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.config.top_p
        )
        
        result = response.choices[0].message.content
        
        # Track token usage
        if hasattr(response, 'usage'):
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
        
        # Log for interpretability if requested
        if log_activations and self.enable_interpretability:
            self.activation_logs.append({
                'prompt': prompt,
                'response': result,
                'timestamp': time.time(),
                'model': self.config.model_name
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
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        
        messages = [{"role": "user", "content": prompt}]
        
        # Request logprobs for interpretability
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.config.top_p,
            logprobs=True,
            top_logprobs=5  # Top 5 alternatives for each token
        )
        
        result = {
            'text': response.choices[0].message.content,
            'logprobs': response.choices[0].logprobs if hasattr(response.choices[0], 'logprobs') else None,
            'model': self.config.model_name,
            'timestamp': time.time()
        }
        
        # Track token usage
        if hasattr(response, 'usage'):
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
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


# Alias for consistency with other API modules
QwenAPI = QwenAPISolver

