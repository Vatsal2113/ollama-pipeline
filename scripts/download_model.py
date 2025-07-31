#!/usr/bin/env python3
import boto3
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", type=str, required=True, help="SageMaker training job name")
    parser.add_argument("--output-dir", type=str, default="./fine-tuned-model", help="Local directory to save the model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Downloading model from training job: {args.job_name}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker')
    
    # Get the S3 model artifact path
    response = sagemaker.describe_training_job(TrainingJobName=args.job_name)
    model_artifact = response['ModelArtifacts']['S3ModelArtifacts']
    
    # Initialize S3 client
    s3 = boto3.client('s3')
    
    # Parse S3 URI
    s3_parts = model_artifact.replace('s3://', '').split('/')
    bucket = s3_parts[0]
    key_prefix = '/'.join(s3_parts[1:])
    
    print(f"Downloading from S3: s3://{bucket}/{key_prefix}")
    
    # Download all files from the S3 path
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                local_path = os.path.join(args.output_dir, os.path.relpath(key, key_prefix))
                
                # Create directory if needed
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                print(f"Downloading {key} to {local_path}")
                s3.download_file(bucket, key, local_path)
    
    print(f"Model downloaded successfully to {args.output_dir}")

if __name__ == "__main__":
    main()