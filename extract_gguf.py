#!/usr/bin/env python3
"""
Extract GGUF files from Ollama models
"""

import os
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_gguf_from_ollama(models_dir="/root/.ollama/models", output_dir="./models"):
    """Extract GGUF files from Ollama models directory"""
    logger.info(f"Extracting GGUF files from {models_dir} to {output_dir}")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Find all GGUF files in Ollama models directory
    gguf_count = 0
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith(".gguf"):
                source_path = os.path.join(root, file)
                target_path = os.path.join(output_dir, file)
                
                logger.info(f"Copying {source_path} to {target_path}")
                shutil.copy(source_path, target_path)
                gguf_count += 1
    
    # Check if we found any GGUF files
    if gguf_count == 0:
        logger.warning("No GGUF files found in Ollama models directory.")
        logger.info(f"Contents of {models_dir}:")
        for root, dirs, files in os.walk(models_dir):
            logger.info(f"Directory: {root}")
            for file in files:
                logger.info(f"  - {file}")
    else:
        logger.info(f"Extracted {gguf_count} GGUF files.")
        
    # List extracted models
    extracted = list(Path(output_dir).glob("*.gguf"))
    logger.info(f"Extracted models in {output_dir}:")
    for model in extracted:
        logger.info(f"  - {model.name}")
    
    return extracted

if __name__ == "__main__":
    extract_gguf_from_ollama()