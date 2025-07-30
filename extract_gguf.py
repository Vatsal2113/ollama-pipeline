#!/usr/bin/env python3
"""
Extract GGUF files from Ollama models
"""

import os
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_gguf_from_ollama(models_dir="/root/.ollama/models", output_dir="./models"):
    """Extract GGUF files from Ollama models directory"""
    logger.info(f"Extracting GGUF files from {models_dir} to {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Find all GGUF files in Ollama models directory
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith(".gguf"):
                source_path = os.path.join(root, file)
                target_path = os.path.join(output_dir, file)
                
                logger.info(f"Copying {source_path} to {target_path}")
                shutil.copy(source_path, target_path)
    
    # List extracted models
    extracted = list(Path(output_dir).glob("*.gguf"))
    logger.info(f"Extracted {len(extracted)} GGUF files:")
    for model in extracted:
        logger.info(f"  - {model.name}")
    
    return extracted

if __name__ == "__main__":
    extract_gguf_from_ollama()