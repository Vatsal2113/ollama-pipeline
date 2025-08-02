#!/usr/bin/env python3
import json
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save evaluation results")
    parser.add_argument("--test-samples", type=int, default=10, help="Number of test samples to evaluate")
    return parser.parse_args()

def evaluate_model(model, tokenizer, test_dataset, num_samples):
    results = []
    
    for i in range(min(num_samples, len(test_dataset))):
        sample = test_dataset[i]
        prompt = sample.get("prompt", sample.get("instruction", ""))
        
        # Get reference response if available
        reference = sample.get("response", sample.get("output", ""))
        
        # Generate model response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9
        )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Record result
        results.append({
            "prompt": prompt,
            "model_response": response,
            "reference_response": reference,
        })
    
    # Calculate basic metrics (can be expanded)
    avg_response_length = np.mean([len(r["model_response"].split()) for r in results])
    
    metrics = {
        "avg_response_length": float(avg_response_length),
        "num_samples_evaluated": len(results)
    }
    
    return {
        "samples": results,
        "metrics": metrics
    }

def main():
    args = parse_args()
    
    print(f"Loading model from: {args.model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    print("Loading evaluation dataset")
    try:
        # Try to load a custom test dataset (if available)
        test_dataset = load_dataset("json", data_files="data/test_samples.json", split="train")
    except:
        # Fall back to a public dataset
        test_dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    print(f"Evaluating model on {args.test_samples} samples")
    results = evaluate_model(model, tokenizer, test_dataset, args.test_samples)
    
    print(f"Saving evaluation results to: {args.output_file}")
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main()