#!/usr/bin/env python3
"""
Verify Control Generation Results
---------------------------------
This script verifies whether the mazes generated in the control experiments
correctly match the requested shapes.

It uses the existing verification logic from the main experiment to check:
1. If the generated maze forms the correct shape
2. If there's a valid path from P to G
3. Structural similarity with existing shape templates
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import re
from collections import defaultdict
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import verification functions
from core.solution_verifier import is_correct_generate, parse_coordinate_maze, encode_maze
from core.maze_generator import SHAPES, all_square_path_mazes, all_cross_path_mazes
from core.maze_generator import all_spiral_path_mazes, all_triangle_path_mazes
from core.maze_generator import all_C_path_mazes, all_Z_path_mazes
from core.maze_generator import init_all_start_end, WALL, PATH, POS, END
from core.prompt_builder import encode_coordinate_list_maze, encode_standard_matrix_maze

# Load environment variables from .env file
load_dotenv()

# Import APIs for intelligent parsing
import os
try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    print("Warning: anthropic library not available. Claude-based verification will be skipped.")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai library not available. GPT-based verification will be skipped.")

# Constants
SIZES = [(5, 5), (7, 7)]
TRIALS = 10


class ControlGenerationVerifier:
    """Class to verify control generation experiments."""
    
    def __init__(self, base_dir):
        """
        Initialize the verifier.
        
        Args:
            base_dir: Base directory containing experiment results
        """
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "data" / "control_experiment_results" / "direct_generation"
        self.results = defaultdict(lambda: defaultdict(list))
        
        # Create extracted mazes directory
        self.extracted_dir = self.base_dir / "analysis_results" / "extracted_mazes"
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different extraction methods
        (self.extracted_dir / "original_extraction").mkdir(exist_ok=True)
        (self.extracted_dir / "gpt_extraction").mkdir(exist_ok=True)
        (self.extracted_dir / "parsing_logs").mkdir(exist_ok=True)
        
    def save_extracted_maze(self, extracted_maze, extraction_method, model_name, encoding_type, size_str, shape, trial, metadata=None):
        """
        Save an extracted maze to file.
        
        Args:
            extracted_maze: The extracted maze text
            extraction_method: 'original' or 'claude'
            model_name: Name of the model
            encoding_type: 'matrix' or 'coordinate'
            size_str: Size string like '5x5'
            shape: Shape name
            trial: Trial number
            metadata: Additional metadata to save
        """
        if extracted_maze is None:
            return None
            
        # Create method-specific directory structure
        method_dir = self.extracted_dir / f"{extraction_method}_extraction"
        model_dir = method_dir / model_name
        encoding_dir = model_dir / encoding_type
        size_dir = encoding_dir / size_str
        shape_dir = size_dir / shape
        
        # Create directories
        shape_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the extracted maze
        maze_file = shape_dir / f"trial_{trial}_maze.txt"
        with open(maze_file, 'w') as f:
            f.write(extracted_maze)
        
        # Save metadata if provided
        if metadata:
            metadata_file = shape_dir / f"trial_{trial}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
                
        return str(maze_file)
        
    def extract_maze_from_response(self, response, encoding_type):
        """
        Extract the generated maze from model response.
        
        Args:
            response: Model's response text
            encoding_type: 'matrix' or 'coordinate'
            
        Returns:
            str: Extracted maze or None if not found
        """
        if encoding_type == "matrix":
            # Look for matrix format maze
            lines = response.strip().split('\n')
            maze_lines = []
            in_maze = False
            
            for line in lines:
                # Check if line looks like a maze row
                if re.match(r'^[01PGpg\s]+$', line.strip()) and len(line.strip()) > 0:
                    # Normalize to uppercase
                    normalized = line.strip().upper()
                    if any(char in normalized for char in ['0', '1', 'P', 'G']):
                        maze_lines.append(normalized)
                        in_maze = True
                elif in_maze and not re.match(r'^[01PGpg\s]+$', line.strip()):
                    # End of maze
                    break
            
            if maze_lines:
                return '\n'.join(maze_lines)
                
        elif encoding_type == "coordinate":
            # Look for coordinate format (with or without **bold** formatting)
            walls_match = re.search(r'\*\*Walls:\*\*\s*([\d\s,()]+)', response, re.IGNORECASE) or \
                         re.search(r'Walls:\s*([\d\s,()]+)', response, re.IGNORECASE)
            empty_match = re.search(r'\*\*Empty:\*\*\s*([\d\s,()]+)', response, re.IGNORECASE) or \
                         re.search(r'Empty:\s*([\d\s,()]+)', response, re.IGNORECASE)
            player_match = re.search(r'\*\*Player\s*(?:position)?:\*\*\s*\((\d+),\s*(\d+)\)', response, re.IGNORECASE) or \
                          re.search(r'Player\s*(?:position)?:\s*\((\d+),\s*(\d+)\)', response, re.IGNORECASE)
            goal_match = re.search(r'\*\*Goal:\*\*\s*\((\d+),\s*(\d+)\)', response, re.IGNORECASE) or \
                        re.search(r'Goal:\s*\((\d+),\s*(\d+)\)', response, re.IGNORECASE)
            
            if walls_match and empty_match and player_match and goal_match:
                maze_text = f"Walls: {walls_match.group(1)}\n"
                maze_text += f"Empty: {empty_match.group(1)}\n"
                maze_text += f"Player position: ({player_match.group(1)}, {player_match.group(2)})\n"
                maze_text += f"Goal: ({goal_match.group(1)}, {goal_match.group(2)})\n"
                return maze_text
        
        return None
    
    def parse_matrix_maze(self, maze_text):
        """
        Parse a matrix format maze into numpy array.
        
        Args:
            maze_text: String representation of maze
            
        Returns:
            np.ndarray or None if parsing fails
        """
        try:
            lines = maze_text.strip().split('\n')
            if not lines:
                return None
                
            # Determine size from first line
            first_row = lines[0].strip().split()
            width = len(first_row)
            height = len(lines)
            
            maze = np.zeros((height, width), dtype=int)
            
            for i, line in enumerate(lines):
                elements = line.strip().split()
                if len(elements) != width:
                    return None  # Invalid maze
                    
                for j, elem in enumerate(elements):
                    if elem == '1':
                        maze[i, j] = WALL
                    elif elem == '0':
                        maze[i, j] = PATH
                    elif elem.upper() == 'P':
                        maze[i, j] = POS
                    elif elem.upper() == 'G':
                        maze[i, j] = END
                    else:
                        return None  # Invalid character
                        
            return maze
            
        except Exception as e:
            print(f"Error parsing matrix maze: {e}")
            return None
    
    def is_correct_generate_control(self, response, encoding_type, size, shape):
        """
        Modified version of is_correct_generate for control experiments.
        Unlike the original, this doesn't exclude any "original" maze since none exists in control experiments.
        
        Args:
            response: Model's raw response text
            encoding_type: 'matrix' or 'coord_list'
            size: Tuple (height, width)
            shape: Expected shape name
            
        Returns:
            bool: True if any valid template for this shape appears in response
        """
        try:
            # Create function mapping for shape generators
            shape_functions = {
                "square": all_square_path_mazes,
                "cross": all_cross_path_mazes,
                "spiral": all_spiral_path_mazes,
                "triangle": all_triangle_path_mazes,
                "C": all_C_path_mazes,
                "Z": all_Z_path_mazes
            }
            
            if shape not in shape_functions:
                print(f"Unknown shape: {shape}")
                return False
            
            # Generate all valid mazes for this shape
            all_mazes = shape_functions[shape](size)
            all_mazes_and_paths = []
            for maze in all_mazes:
                all_mazes_and_paths.extend(init_all_start_end(maze))
            
            # Encode all templates
            all_maze_encodings = set()
            for maze in all_mazes_and_paths:
                all_maze_encodings.add(encode_maze(maze, encoding_type))
            
            # Check if ANY template encoding appears in the response
            # (No removal step - this is the key difference from regular experiments)
            for maze_encoding in all_maze_encodings:
                if maze_encoding in response:
                    return True
            return False
            
        except Exception as e:
            print(f"Error in is_correct_generate_control for {shape}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def gpt_extract_and_verify(self, response, encoding_type, size, shape):
        """
        Use GPT-4o to intelligently extract and parse the generated maze,
        then apply canonical verification algorithm.
        
        Args:
            response: Model's raw response text
            encoding_type: 'matrix' or 'coord_list'
            size: Tuple (height, width)
            shape: Expected shape name
            
        Returns:
            dict: Results including extraction success and verification outcome
        """
        if not OPENAI_AVAILABLE:
            return {
                "success": False,
                "error": "OpenAI API not available",
                "extracted_maze": None,
                "verification_result": False
            }
        
        try:
            # Get API key
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return {
                    "success": False,
                    "error": "OPENAI_API_KEY not set",
                    "extracted_maze": None,
                    "verification_result": False
                }
            
            # Initialize OpenAI client
            client = openai.OpenAI(api_key=api_key)
            
            # Create extraction prompt based on encoding type
            if encoding_type == "matrix":
                extraction_prompt = f"""
You are a maze extraction expert. Extract the maze from this response and convert it to the exact canonical format.

The response contains a maze generation attempt. Extract the maze and format it EXACTLY as follows:

Matrix format (space-separated values in a grid):
1 1 1 0 1
1 P 0 0 1
1 0 1 G 1
1 0 0 0 1
1 1 1 1 1

Where:
- 1 = wall
- 0 = empty path
- P = player/start position  
- G = goal/end position

The maze should be exactly {size[0]}x{size[1]} in size.

Response to extract from:
{response[:2000]}

Return ONLY the maze grid, nothing else. If no valid maze can be extracted, return "NO_MAZE_FOUND".
"""
            else:  # coordinate encoding
                extraction_prompt = f"""
You are a maze extraction expert. Extract the maze from this response and convert it to the exact canonical format.

The response contains a maze generation attempt. Extract the maze and format it EXACTLY as follows:

**Walls:** (0,0), (0,1), (0,2), (0,4), (1,0), (1,2), (1,4)
**Empty:** (1,1), (1,3), (2,1), (2,2), (2,3)
**Player position:** (1,1)
**Goal:** (2,3)

The maze should be exactly {size[0]}x{size[1]} in size (coordinates from 0 to {size[0]-1}, 0 to {size[1]-1}).

Response to extract from:
{response[:2000]}

Return ONLY the four lines above with the exact **bold** formatting, nothing else. If no valid maze can be extracted, return "NO_MAZE_FOUND".
"""
            
            # Make API call to GPT-5 with rate limiting protection
            import time
            max_retries = 3
            retry_delay = 1  # Start with 1 second delay
            
            for attempt in range(max_retries):
                try:
                    gpt_response = client.chat.completions.create(
                        model="gpt-4o",
                        max_tokens=1000,
                        temperature=0,
                        messages=[{"role": "user", "content": extraction_prompt}]
                    )
                    break  # Success, exit retry loop
                    
                except Exception as api_error:
                    print(f"❌ GPT-4o API error: {str(api_error)}")
                    if "rate_limit" in str(api_error).lower() and attempt < max_retries - 1:
                        # Rate limited, wait and retry
                        print(f"⏳ Rate limited, waiting {retry_delay}s before retry...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        # Not rate limit or final attempt, re-raise
                        raise api_error
            
            extracted_text = gpt_response.choices[0].message.content.strip()
            
            if extracted_text == "NO_MAZE_FOUND":
                return {
                    "success": False,
                    "error": "GPT could not extract maze",
                    "extracted_maze": None,
                    "verification_result": False
                }
            
            if not extracted_text:
                return {
                    "success": False,
                    "error": "GPT returned empty response",
                    "extracted_maze": None,
                    "verification_result": False
                }
            
            # Parse the extracted maze using our existing logic
            if encoding_type == "matrix":
                # Parse matrix format
                maze = self.parse_matrix_maze(extracted_text)
                if maze is None:
                    return {
                        "success": False,
                        "error": "Failed to parse GPT-extracted matrix maze",
                        "extracted_maze": extracted_text,
                        "verification_result": False
                    }
            else:  # coordinate format
                # Parse coordinate format - normalize GPT-4o output to expected format
                try:
                    # Normalize the extracted text to match parse_coordinate_maze expectations
                    normalized_text = extracted_text
                    # Remove **bold** formatting
                    normalized_text = re.sub(r'\*\*(.*?)\*\*', r'\1', normalized_text)
                    # Replace "Player position:" with "Start:"
                    normalized_text = re.sub(r'Player position:', 'Start:', normalized_text)
                    # Replace "Goal:" with "End:"
                    normalized_text = re.sub(r'Goal:', 'End:', normalized_text)
                    # Add spaces after commas in coordinates: (0,1) -> (0, 1)
                    normalized_text = re.sub(r'\((\d+),(\d+)\)', r'(\1, \2)', normalized_text)
                    
                    maze = parse_coordinate_maze(normalized_text, size)
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Failed to parse GPT-extracted coordinate maze: {e}",
                        "extracted_maze": extracted_text,
                        "verification_result": False
                    }
            
            # Now verify the parsed maze using canonical verification
            # Encode the maze to canonical format
            if encoding_type == "matrix":
                canonical_encoding = encode_standard_matrix_maze(maze)
            else:
                canonical_encoding = encode_coordinate_list_maze(maze)
            
            # Use the original verification logic (checking against templates)
            verification_result = self.is_correct_generate_control(
                canonical_encoding, encoding_type, size, shape
            )
            
            return {
                "success": True,
                "error": None,
                "extracted_maze": extracted_text,
                "parsed_maze": maze,
                "canonical_encoding": canonical_encoding,
                "verification_result": verification_result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"GPT extraction failed: {str(e)}",
                "extracted_maze": None,
                "verification_result": False
            }
    
    def verify_path_exists(self, maze):
        """
        Verify if there's a valid path from P to G in the maze.
        
        Args:
            maze: numpy array representation
            
        Returns:
            bool: True if valid path exists
        """
        try:
            # Find start and end positions
            start_pos = np.argwhere(maze == POS)
            end_pos = np.argwhere(maze == END)
            
            if len(start_pos) != 1 or len(end_pos) != 1:
                return False
                
            start = tuple(start_pos[0])
            end = tuple(end_pos[0])
            
            # Simple BFS to check path existence
            from collections import deque
            
            queue = deque([start])
            visited = set([start])
            
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
            
            while queue:
                current = queue.popleft()
                
                if current == end:
                    return True
                    
                for dx, dy in directions:
                    next_pos = (current[0] + dx, current[1] + dy)
                    
                    # Check bounds
                    if (0 <= next_pos[0] < maze.shape[0] and 
                        0 <= next_pos[1] < maze.shape[1] and
                        next_pos not in visited and
                        maze[next_pos] != WALL):
                        
                        visited.add(next_pos)
                        queue.append(next_pos)
                        
            return False
            
        except Exception:
            return False
    
    def verify_single_result(self, result_file, encoding_type, size, shape, show_progress=True):
        """
        Verify a single experimental result.
        
        Args:
            result_file: Path to result file
            encoding_type: 'matrix' or 'coordinate'
            size: Tuple (height, width)
            shape: Expected shape
            show_progress: Whether to show progress messages
            
        Returns:
            dict: Verification results
        """
        verification = {
            "file": str(result_file),
            "shape_correct": False,
            "has_valid_path": False,
            "maze_extracted": False,
            "error": None
        }
        
        if show_progress:
            file_name = str(result_file).split('/')[-1]
            print(f"    📄 Processing {file_name}...", end=" ", flush=True)
        
        try:
            # Read result
            with open(result_file, 'r') as f:
                content = f.read()
            
            # Check if shape is correct using modified logic for control experiments
            # Unlike regular experiments, we don't exclude any "original" maze since none exists
            if show_progress:
                print("🔍", end="", flush=True)
            verification["shape_correct"] = self.is_correct_generate_control(
                content,  # Use raw response directly
                encoding_type if encoding_type == "matrix" else "coord_list",
                size, 
                shape
            )
            
            # ADDITIONAL VERIFICATION: Use GPT-5 for intelligent extraction
            if show_progress:
                print("🤖", end="", flush=True)
            gpt_result = self.gpt_extract_and_verify(
                content, 
                encoding_type if encoding_type == "matrix" else "coord_list",
                size, 
                shape
            )
            
            verification["gpt_extraction"] = {
                "success": gpt_result["success"],
                "error": gpt_result["error"],
                "shape_correct": gpt_result["verification_result"]
            }
            
            if gpt_result["success"]:
                verification["gpt_extracted_maze"] = gpt_result["extracted_maze"]
                verification["gpt_canonical_encoding"] = gpt_result.get("canonical_encoding", None)
                
                # Save GPT-extracted maze
                model_name = str(result_file).split('/')[-4]  # Extract model name from path
                size_str = f"{size[0]}x{size[1]}"
                trial = int(str(result_file).split('_')[-1].split('.')[0])  # Extract trial number
                
                gpt_maze_file = self.save_extracted_maze(
                    gpt_result["extracted_maze"],
                    "gpt",
                    model_name,
                    encoding_type,
                    size_str,
                    shape,
                    trial,
                    {
                        "original_file": str(result_file),
                        "canonical_encoding": gpt_result.get("canonical_encoding", None),
                        "shape_correct": gpt_result["verification_result"],
                        "parsed_maze": gpt_result.get("parsed_maze", None) is not None
                    }
                )
                verification["gpt_maze_file"] = gpt_maze_file
            
            # Try to extract maze for additional analysis (path validation)
            if show_progress:
                print("🧩", end="", flush=True)
            maze_text = self.extract_maze_from_response(content, encoding_type)
            
            if maze_text:
                verification["maze_extracted"] = True
                verification["maze_text"] = maze_text
                if show_progress:
                    print("💾", end="", flush=True)
                
                # Save original extracted maze
                model_name = str(result_file).split('/')[-4]  # Extract model name from path
                size_str = f"{size[0]}x{size[1]}"
                trial = int(str(result_file).split('_')[-1].split('.')[0])  # Extract trial number
                
                original_maze_file = self.save_extracted_maze(
                    maze_text,
                    "original",
                    model_name,
                    encoding_type,
                    size_str,
                    shape,
                    trial,
                    {
                        "original_file": str(result_file),
                        "extraction_method": "regex_based",
                        "shape_correct": verification["shape_correct"]
                    }
                )
                verification["original_maze_file"] = original_maze_file
                
                # Parse maze based on encoding
                maze = None
                if encoding_type == "matrix":
                    maze = self.parse_matrix_maze(maze_text)
                else:  # coordinate
                    try:
                        maze = parse_coordinate_maze(maze_text, size)
                    except Exception as e:
                        print(f"Parse error for {result_file}: {e}")
                
                # Check if valid path exists (only if we could parse the maze)
                if maze is not None:
                    verification["has_valid_path"] = self.verify_path_exists(maze)
                    verification["maze"] = maze
                else:
                    verification["has_valid_path"] = False
            else:
                verification["maze_extracted"] = False
                verification["has_valid_path"] = False
            
        except Exception as e:
            verification["error"] = f"Verification error: {str(e)}"
            if show_progress:
                print("❌", end="", flush=True)
            
        # Show completion status
        if show_progress:
            status_symbols = []
            if verification["shape_correct"]:
                status_symbols.append("✓")
            else:
                status_symbols.append("✗")
            
            if verification.get("gpt_extraction", {}).get("shape_correct", False):
                status_symbols.append("GPT✓")
            elif verification.get("gpt_extraction", {}).get("success", False):
                status_symbols.append("GPT✗")
            else:
                status_symbols.append("GPT❌")
                
            if verification["has_valid_path"]:
                status_symbols.append("🗺️")
            
            print(f" {' '.join(status_symbols)}")
            
        return verification
    
    def verify_all_results(self):
        """
        Verify all control generation results.
        
        Returns:
            dict: Summary of verification results
        """
        print("Starting verification of control generation results...")
        print("Legend: 📄=Processing 🔍=Shape check 🤖=GPT extraction 🧩=Original extraction 💾=Saving")
        print("Results: ✓=Correct ✗=Incorrect ❌=Failed GPT✓=GPT correct 🗺️=Valid path\n")
        
        all_results = []
        summary = defaultdict(lambda: defaultdict(int))
        total_files = 0
        processed_files = 0
        
        # Count total files first
        for results_subdir in self.results_dir.iterdir():
            if not results_subdir.is_dir():
                continue
            for size_str in ["5x5", "7x7"]:
                size_dir = results_subdir / size_str
                if not size_dir.exists():
                    continue
                for shape in SHAPES:
                    shape_dir = size_dir / shape
                    if not shape_dir.exists():
                        continue
                    for trial in range(TRIALS):
                        result_file = shape_dir / f"trial_{trial}.txt"
                        if result_file.exists():
                            total_files += 1
        
        print(f"Total files to process: {total_files}\n")
        
        # Process each results directory
        for results_subdir in self.results_dir.iterdir():
            if not results_subdir.is_dir():
                continue
                
            # Determine encoding type from directory name
            if "matrix" in results_subdir.name:
                encoding_type = "matrix"
            elif "coord" in results_subdir.name:
                encoding_type = "coordinate"
            else:
                continue
                
            model_name = results_subdir.name
            print(f"\n🤖 {model_name} ({encoding_type})")
            
            # Process each size
            for size_str in ["5x5", "7x7"]:
                size_tuple = (int(size_str.split('x')[0]), int(size_str.split('x')[1]))
                size_dir = results_subdir / size_str
                
                if not size_dir.exists():
                    continue
                    
                # Process each shape
                for shape in SHAPES:
                    shape_dir = size_dir / shape
                    
                    if not shape_dir.exists():
                        continue
                        
                    print(f"  🔷 {size_str} {shape}:")
                        
                    # Process each trial
                    for trial in range(TRIALS):
                        result_file = shape_dir / f"trial_{trial}.txt"
                        
                        if not result_file.exists():
                            continue
                            
                        # Verify this result
                        verification = self.verify_single_result(
                            result_file, encoding_type, size_tuple, shape, show_progress=True
                        )
                        
                        processed_files += 1
                        
                        # Add small delay to prevent rate limiting
                        import time
                        time.sleep(0.1)  # 100ms delay between files
                        
                        # Add metadata
                        verification["model"] = model_name
                        verification["encoding"] = encoding_type
                        verification["size"] = size_str
                        verification["shape"] = shape
                        verification["trial"] = trial
                        
                        all_results.append(verification)
                        
                        # Update summary
                        key = f"{model_name}_{encoding_type}"
                        summary[key]["total"] += 1
                        
                        if verification["shape_correct"]:
                            summary[key]["shape_correct"] += 1
                        if verification["has_valid_path"]:
                            summary[key]["path_valid"] += 1
                        if verification["maze_extracted"]:
                            summary[key]["extracted"] += 1
                        
                        # GPT-based verification results
                        if verification.get("gpt_extraction", {}).get("success", False):
                            summary[key]["gpt_extracted"] = summary[key].get("gpt_extracted", 0) + 1
                        if verification.get("gpt_extraction", {}).get("shape_correct", False):
                            summary[key]["gpt_shape_correct"] = summary[key].get("gpt_shape_correct", 0) + 1
                        
                        # Per-shape summary
                        shape_key = f"{key}_{shape}"
                        summary[shape_key]["total"] += 1
                        if verification["shape_correct"]:
                            summary[shape_key]["shape_correct"] += 1
                        if verification.get("gpt_extraction", {}).get("shape_correct", False):
                            summary[shape_key]["gpt_shape_correct"] = summary[shape_key].get("gpt_shape_correct", 0) + 1
                            
                        # Per-size summary
                        size_key = f"{key}_{size_str}"
                        summary[size_key]["total"] += 1
                        if verification["shape_correct"]:
                            summary[size_key]["shape_correct"] += 1
                        if verification.get("gpt_extraction", {}).get("shape_correct", False):
                            summary[size_key]["gpt_shape_correct"] = summary[size_key].get("gpt_shape_correct", 0) + 1
                        
                        # Show overall progress periodically
                        if processed_files % 20 == 0:
                            percentage = (processed_files / total_files) * 100
                            print(f"\n    📈 Progress: {processed_files}/{total_files} ({percentage:.1f}%) completed")
                    
                    # Show shape completion
                    shape_files = len([f for f in shape_dir.glob('*.txt')])
                    print(f"    ✓ Completed {shape} ({shape_files} files)")
        
        print(f"\n✓ All verification complete! Processed {processed_files}/{total_files} files\n")
        
        # Save detailed results
        results_file = self.results_dir / "verification_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Save summary
        summary_file = self.results_dir / "verification_summary.json"
        summary_dict = {k: dict(v) for k, v in summary.items()}
        with open(summary_file, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        
        # Print summary
        self.print_summary(summary_dict)
        
        return summary_dict
    
    def print_summary(self, summary):
        """Print a formatted summary of results."""
        print("\n" + "="*80)
        print("VERIFICATION SUMMARY")
        print("="*80)
        
        # Overall results by model
        print("\nOVERALL RESULTS BY MODEL:")
        print("-"*50)
        
        models_seen = set()
        for key in summary:
            if "_" in key and not any(shape in key for shape in SHAPES) and not any(size in key for size in ["5x5", "7x7"]):
                models_seen.add(key)
        
        for model_key in sorted(models_seen):
            data = summary[model_key]
            total = data.get("total", 0)
            if total == 0:
                continue
                
            shape_correct = data.get("shape_correct", 0)
            path_valid = data.get("path_valid", 0)
            extracted = data.get("extracted", 0)
            gpt_extracted = data.get("gpt_extracted", 0)
            gpt_shape_correct = data.get("gpt_shape_correct", 0)
            
            print(f"\n{model_key}:")
            print(f"  Total trials: {total}")
            print(f"  Original verification:")
            print(f"    Mazes extracted: {extracted}/{total} ({100*extracted/total:.1f}%)")
            print(f"    Shape correct: {shape_correct}/{total} ({100*shape_correct/total:.1f}%)")
            print(f"    Valid path: {path_valid}/{total} ({100*path_valid/total:.1f}%)")
            print(f"  GPT-based verification:")
            print(f"    Extracted: {gpt_extracted}/{total} ({100*gpt_extracted/total:.1f}%)")
            print(f"    Shape correct: {gpt_shape_correct}/{total} ({100*gpt_shape_correct/total:.1f}%)")
        
        # Results by shape
        print("\n\nRESULTS BY SHAPE:")
        print("-"*50)
        
        for shape in SHAPES:
            print(f"\n{shape.upper()}:")
            for model_key in sorted(models_seen):
                shape_key = f"{model_key}_{shape}"
                if shape_key in summary:
                    data = summary[shape_key]
                    total = data.get("total", 0)
                    if total > 0:
                        original_correct = data.get("shape_correct", 0)
                        gpt_correct = data.get("gpt_shape_correct", 0)
                        print(f"  {model_key}: Original={original_correct}/{total} ({100*original_correct/total:.1f}%), GPT={gpt_correct}/{total} ({100*gpt_correct/total:.1f}%)")
        
        # Results by size
        print("\n\nRESULTS BY SIZE:")
        print("-"*50)
        
        for size in ["5x5", "7x7"]:
            print(f"\n{size}:")
            for model_key in sorted(models_seen):
                size_key = f"{model_key}_{size}"
                if size_key in summary:
                    data = summary[size_key]
                    total = data.get("total", 0)
                    if total > 0:
                        original_correct = data.get("shape_correct", 0)
                        gpt_correct = data.get("gpt_shape_correct", 0)
                        print(f"  {model_key}: Original={original_correct}/{total} ({100*original_correct/total:.1f}%), GPT={gpt_correct}/{total} ({100*gpt_correct/total:.1f}%)")
        
        print("\n" + "="*80)
        print(f"Full results saved to: {self.results_dir / 'verification_results.json'}")
        print(f"Summary saved to: {self.results_dir / 'verification_summary.json'}")
        print(f"Extracted mazes saved to: {self.extracted_dir}")
        print(f"  - Original extraction: {self.extracted_dir / 'original_extraction'}")
        print(f"  - GPT extraction: {self.extracted_dir / 'gpt_extraction'}")
        print(f"  - Parsing logs: {self.extracted_dir / 'parsing_logs'}")
        print("\nExtracted mazes are organized by: method/model/encoding/size/shape/trial_X_maze.txt")


def main():
    """Main function to verify control generation results."""
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Check if results exist
    results_dir = project_root / "data" / "control_experiment_results" / "direct_generation"
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Please run run_control_experiments.py first")
        return
    
    # Create verifier and run verification
    verifier = ControlGenerationVerifier(project_root)
    verifier.verify_all_results()


if __name__ == "__main__":
    main()
