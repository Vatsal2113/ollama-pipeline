#!/usr/bin/env python3
import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=str, required=True, help="Evaluation results JSON file")
    parser.add_argument("--output-file", type=str, required=True, help="Output markdown report file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading evaluation results from: {args.eval_file}")
    with open(args.eval_file, "r") as f:
        results = json.load(f)
    
    metrics = results["metrics"]
    samples = results["samples"]
    
    # Generate markdown report
    report = [
        "# Model Evaluation Report",
        "",
        "## Model Performance Metrics",
        "",
        f"- **Average Response Length**: {metrics['avg_response_length']:.1f} words",
        f"- **Number of Samples Evaluated**: {metrics['num_samples_evaluated']}",
        "",
        "## Sample Responses",
        "",
    ]
    
    # Add sample table
    report.append("| # | Prompt | Model Response |")
    report.append("| --- | --- | --- |")
    
    # Include up to 5 samples
    for i, sample in enumerate(samples[:5]):
        prompt = sample["prompt"].replace("\n", " ").strip()
        if len(prompt) > 50:
            prompt = prompt[:50] + "..."
            
        response = sample["model_response"].replace("\n", " ").strip()
        if len(response) > 100:
            response = response[:100] + "..."
            
        report.append(f"| {i+1} | {prompt} | {response} |")
    
    report.append("")
    report.append(f"S3 path: s3://${{BUCKET_NAME}}/models/${{MODEL_NAME}}/latest/")
    
    print(f"Writing report to: {args.output_file}")
    with open(args.output_file, "w") as f:
        f.write("\n".join(report))
    
    print("Report generated successfully!")

if __name__ == "__main__":
    main()