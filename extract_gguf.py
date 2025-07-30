#!/usr/bin/env python3

import os
import shutil
import argparse
import json
import subprocess
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Extract GGUF files from Ollama models")
    parser.add_argument("--output", default="./models", 
                        help="Output directory for extracted files")
    return parser.parse_args()

def find_and_copy_gguf_files(ollama_dir, output_dir):
    """Find and copy all GGUF files from Ollama directory to output directory"""
    # Default Ollama model path
    ollama_path = os.path.expanduser("~/.ollama/models")
    
    if not os.path.exists(ollama_path):
        print(f"Error: Ollama models directory not found at {ollama_path}")
        return False
        
    print(f"Searching for GGUF files in {ollama_path}...")
    
    # Find all .gguf files
    gguf_files = []
    for root, _, files in os.walk(ollama_path):
        for file in files:
            if file.endswith(".gguf"):
                gguf_files.append(os.path.join(root, file))
    
    if not gguf_files:
        print("No GGUF files found!")
        return False
    
    # Copy files to output directory
    print(f"Found {len(gguf_files)} GGUF files")
    os.makedirs(output_dir, exist_ok=True)
    
    for file_path in gguf_files:
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(output_dir, file_name)
        print(f"Copying {file_name} to {dest_path}")
        shutil.copy2(file_path, dest_path)
    
    print(f"Successfully extracted {len(gguf_files)} GGUF files to {output_dir}")
    return True

def get_ollama_models():
    """Get list of available models from Ollama API"""
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True, text=True, check=True
        )
        data = json.loads(result.stdout)
        return data.get("models", [])
    except Exception as e:
        print(f"Error getting models from Ollama API: {e}")
        return []

def main():
    args = parse_args()
    
    # Get models from Ollama API
    models = get_ollama_models()
    if models:
        print(f"Found {len(models)} models in Ollama")
        for model in models:
            print(f" - {model.get('name')}")
    
    # Extract GGUF files
    success = find_and_copy_gguf_files(os.path.expanduser("~/.ollama"), args.output)
    
    if success:
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())