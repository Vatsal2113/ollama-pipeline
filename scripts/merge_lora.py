#!/usr/bin/env python3
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, required=True, help="Base model name or path")
    parser.add_argument("--lora-model", type=str, required=True, help="Path to LoRA adapter model")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for merged model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading LoRA model from: {args.lora_model}")
    model = PeftModel.from_pretrained(base_model, args.lora_model)
    
    print("Merging weights...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    
    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Model merge completed successfully!")

if __name__ == "__main__":
    main()