#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with merged model")
    parser.add_argument("--output-file", type=str, required=True, help="Output GGUF file path")
    parser.add_argument("--outtype", type=str, default="q4_k_m", help="Quantization type (e.g., q4_k_m, q5_k_m)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if llama.cpp repo exists, clone if not
    if not os.path.exists("llama.cpp"):
        print("Cloning llama.cpp repository...")
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"], check=True)
        
        # Build llama.cpp
        print("Building llama.cpp...")
        os.chdir("llama.cpp")
        subprocess.run(["make"], check=True)
        os.chdir("..")
    
    # Convert to GGUF format
    print(f"Converting model to GGUF format: {args.output_file}")
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "python3", "llama.cpp/convert.py", 
        "--outtype", args.outtype,
        "--outfile", args.output_file,
        args.input_dir
    ]
    
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    
    print(f"Conversion complete! GGUF model saved to: {args.output_file}")

if __name__ == "__main__":
    main()