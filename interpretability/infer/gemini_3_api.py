"""
Google Gemini 3 API implementation for maze solving experiments with vision support.
Supports both text and vision-based approaches.
"""

import os
import base64
from PIL import Image
import io
from dotenv import load_dotenv
import google.generativeai as genai
from infer.base_api import BaseAPISolver
from infer.base_vision_api import BaseVisionAPISolver

# Load environment variables
load_dotenv()


class Gemini3APISolver(BaseVisionAPISolver):
    """Gemini 3 API implementation for maze solving with text and vision support."""
    
    def __init__(self, api_key: str = None, model: str = "gemini-1.5-pro"):
        """Initialize Gemini 3 solver with optional model selection."""
        self.model_name = model
        super().__init__(api_key)
    
    def _get_api_key(self) -> str:
        """Get API key from environment variables."""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        return api_key
    
    def _initialize_client(self):
        """Initialize the Gemini client."""
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(self.model_name)
    
    def _make_api_call(self, messages: list, **kwargs) -> str:
        """Make an API call to Gemini and return the response text."""
        # Handle vision messages differently from text-only messages
        if self._is_vision_message(messages):
            return self._make_vision_api_call(messages, **kwargs)
        else:
            return self._make_text_api_call(messages, **kwargs)
    
    def _is_vision_message(self, messages: list) -> bool:
        """Check if any message contains vision content."""
        for message in messages:
            content = message.get('content', '')
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'image':
                        return True
        return False
    
    def _make_vision_api_call(self, messages: list, **kwargs) -> str:
        """Make a vision API call to Gemini with multimodal content."""
        # Get the last user message with vision content
        last_user_message = None
        for message in reversed(messages):
            if message.get('role') == 'user':
                last_user_message = message
                break
        
        if not last_user_message:
            raise ValueError("No user message found for vision API call")
        
        content = last_user_message['content']
        if isinstance(content, list):
            # Convert the multimodal content for Gemini
            gemini_content = []
            for part in content:
                if part.get('type') == 'text':
                    gemini_content.append(part['text'])
                elif part.get('type') == 'image':
                    # Convert base64 image to PIL Image for Gemini
                    image_base64 = part['image_base64']
                    image_data = base64.b64decode(image_base64)
                    image = Image.open(io.BytesIO(image_data))
                    gemini_content.append(image)
            
            response = self.client.generate_content(
                gemini_content,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get('temperature', 0),
                    max_output_tokens=kwargs.get('max_tokens', 1000),
                )
            )
        else:
            # Fallback to text-only
            response = self.client.generate_content(
                content,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get('temperature', 0),
                    max_output_tokens=kwargs.get('max_tokens', 1000),
                )
            )
        
        return response.text
    
    def _make_text_api_call(self, messages: list, **kwargs) -> str:
        """Make a text-only API call to Gemini."""
        # Convert messages to Gemini format
        prompt = self._convert_messages_to_prompt(messages)
        
        response = self.client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=kwargs.get('temperature', 0),
                max_output_tokens=kwargs.get('max_tokens', 1000),
            )
        )
        return response.text
    
    def _convert_messages_to_prompt(self, messages: list) -> str:
        """Convert OpenAI-style messages to Gemini prompt format."""
        # For text-only messages, concatenate all messages into a single prompt
        prompt_parts = []
        
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            
            # Handle string content
            if isinstance(content, str):
                if role == 'user':
                    prompt_parts.append(f"User: {content}")
                elif role == 'assistant':
                    prompt_parts.append(f"Assistant: {content}")
                elif role == 'system':
                    prompt_parts.insert(0, f"Context: {content}")
        
        return "\n\n".join(prompt_parts)
    
    def _format_vision_message(self, text: str, image_base64: str) -> list:
        """Format a message with text and image for Gemini's API."""
        return [
            {"type": "text", "text": text},
            {"type": "image", "image_base64": image_base64}
        ]
    
    def _get_output_directory(self, encoding_type: str) -> str:
        """Get the output directory name for results."""
        if encoding_type == "vision":
            return "vision_gemini3_results"
        else:
            return f"{encoding_type}_gemini3_results"


def main():
    """Run experiments with Gemini 3 API."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run maze solving experiments with Gemini 3 models')
    parser.add_argument('--model', type=str, default='gemini-1.5-pro',
                       help='Gemini model to use')
    parser.add_argument('--encoding', type=str, nargs='+', 
                       default=['matrix', 'coord_list', 'vision'],
                       choices=['matrix', 'coord_list', 'vision'],
                       help='Encoding types to use')
    parser.add_argument('--sizes', type=str, nargs='+', default=['5x5', '7x7'],
                       help='Maze sizes to test')
    
    args = parser.parse_args()
    
    # Parse sizes
    sizes = []
    for size_str in args.sizes:
        parts = size_str.split('x')
        sizes.append((int(parts[0]), int(parts[1])))
    
    # Create solver
    solver = Gemini3APISolver(model=args.model)
    
    # Run experiments
    solver.run_all_experiments(
        encoding_types=args.encoding,
        sizes=sizes,
        shapes=None  # Use all shapes
    )


if __name__ == "__main__":
    main()

# Backwards-compatible alias expected by some runners
Gemini3API = Gemini3APISolver
