#!/usr/bin/env python3
"""
Script to quantize finetuned models to GGUF format
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_args():
    parser = argparse.ArgumentParser(description="Quantize HF model to GGUF format")
    parser.add_argument("--input_model", type=str, required=True, help="Path to input model directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for quantized model")
    parser.add_argument("--quantization_level", type=str, default="Q4_K_M", 
                        choices=["Q4_0", "Q4_K_M", "Q5_0", "Q5_K_M", "Q6_K", "Q8_0"], 
                        help="Quantization level")
    return parser.parse_args()

def install_llama_cpp_python():
    logger.info("Installing llama-cpp-python for quantization...")
    subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python[server]"], check=True)

def quantize_model(input_model_path, output_dir, quant_level):
    """Quantize model to GGUF format using llama-cpp-python"""
    try:
        import llama_cpp
        from llama_cpp.model_converter import convert_hf_to_gguf
    except ImportError:
        install_llama_cpp_python()
        import llama_cpp
        from llama_cpp.model_converter import convert_hf_to_gguf
    
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Get model name from input path
    model_name = os.path.basename(os.path.normpath(input_model_path))
    output_path = os.path.join(output_dir, f"{model_name}-{quant_level.lower()}.gguf")
    
    logger.info(f"Converting model from {input_model_path} to GGUF...")
    logger.info(f"Quantization level: {quant_level}")
    logger.info(f"Output path: {output_path}")
    
    # Convert the model
    convert_hf_to_gguf(
        model_path=input_model_path,
        output_path=output_path,
        quantization_level=quant_level
    )
    
    logger.info(f"Model successfully quantized and saved to {output_path}")
    return output_path

def main():
    args = setup_args()
    quantize_model(args.input_model, args.output_dir, args.quantization_level)

if __name__ == "__main__":
    main()