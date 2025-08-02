#!/usr/bin/env python3
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf-path", type=str, required=True, help="Path to GGUF file")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for Modelfile")
    parser.add_argument("--model-name", type=str, required=True, help="Name for the Ollama model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    modelfile_path = os.path.join(args.output_dir, "Modelfile")
    
    # Get the GGUF filename (without the path)
    gguf_filename = os.path.basename(args.gguf_path)
    
    print(f"Creating Modelfile at: {modelfile_path}")
    
    with open(modelfile_path, 'w') as f:
        f.write(f'FROM {gguf_filename}\n\n')
        f.write(f'# Model configuration\n')
        f.write(f'PARAMETER temperature 0.7\n')
        f.write(f'PARAMETER top_p 0.9\n')
        f.write(f'PARAMETER stop "User:"\n')
        f.write(f'PARAMETER stop "Assistant:"\n\n')
        f.write(f'# Instruction template for {args.model_name}\n')
        f.write(f'TEMPLATE """<s>[INST] {{prompt}} [/INST]"""')
    
    print(f"Modelfile created successfully! Use 'ollama create {args.model_name} -f {modelfile_path}' to import.")

if __name__ == "__main__":
    main()