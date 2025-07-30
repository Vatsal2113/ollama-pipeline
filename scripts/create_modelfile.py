#!/usr/bin/env python3
"""
Script to create Modelfile for Ollama from GGUF model
"""

import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_args():
    parser = argparse.ArgumentParser(description="Create Modelfile for Ollama")
    parser.add_argument("--model_name", type=str, required=True, help="Name for the Ollama model")
    parser.add_argument("--gguf_path", type=str, required=True, help="Path to GGUF model file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for Modelfile")
    return parser.parse_args()

def create_modelfile(model_name, gguf_path, output_dir):
    """Create Modelfile for Ollama"""
    # Find GGUF file
    if os.path.isdir(gguf_path):
        gguf_files = list(Path(gguf_path).glob("*.gguf"))
        if not gguf_files:
            raise ValueError(f"No GGUF files found in {gguf_path}")
        gguf_file = gguf_files[0]  # Use the first GGUF file if multiple
    else:
        gguf_file = gguf_path
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Create Modelfile
    modelfile_content = f"""FROM {os.path.basename(gguf_file)}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048
PARAMETER repeat_penalty 1.1

SYSTEM You are a helpful assistant that provides accurate and concise responses.
"""
    
    modelfile_path = os.path.join(output_dir, "Modelfile")
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    # Copy GGUF file to output directory
    import shutil
    shutil.copy(gguf_file, os.path.join(output_dir, os.path.basename(gguf_file)))
    
    logger.info(f"Created Modelfile at {modelfile_path}")
    logger.info(f"Copied GGUF file to {output_dir}")
    
    return modelfile_path

def main():
    args = setup_args()
    create_modelfile(args.model_name, args.gguf_path, args.output_dir)

if __name__ == "__main__":
    main()