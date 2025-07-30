#!/usr/bin/env python3

import os
import boto3
import argparse
import sagemaker
from sagemaker.huggingface import HuggingFace

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune models in SageMaker")
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket name")
    parser.add_argument("--base-model", required=True, help="Base model to finetune")
    parser.add_argument("--training-data", required=True, help="S3 path to training data")
    parser.add_argument("--output-path", required=True, help="S3 path to store output")
    parser.add_argument("--instance-type", default="ml.g4dn.xlarge", help="SageMaker instance type")
    return parser.parse_args()

def setup_sagemaker_session():
    """Set up SageMaker session"""
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    return sagemaker_session, role

def create_finetune_job(args, sagemaker_session, role):
    """Create and start a SageMaker finetuning job"""
    # Define hyperparameters
    hyperparameters = {
        'model_name_or_path': args.base_model,
        'output_dir': '/opt/ml/model',
        'per_device_train_batch_size': '4',
        'gradient_accumulation_steps': '2',
        'learning_rate': '2e-5',
        'num_train_epochs': '3',
        'save_strategy': 'epoch',
    }
    
    # Create HuggingFace estimator
    huggingface_estimator = HuggingFace(
        entry_point='run_clm.py',  # Training script
        source_dir='./scripts',    # Directory containing training scripts
        instance_type=args.instance_type,
        instance_count=1,
        role=role,
        transformers_version='4.26.0',
        pytorch_version='1.13.1',
        py_version='py39',
        hyperparameters=hyperparameters
    )
    
    # Start training job
    huggingface_estimator.fit({
        'train': args.training_data,
    })
    
    # Return model artifacts location
    model_data = huggingface_estimator.model_data
    print(f"Model artifacts saved to: {model_data}")
    return model_data

def main():
    args = parse_args()
    
    # Check if AWS credentials are available
    if not all([os.environ.get('AWS_ACCESS_KEY_ID'), 
                os.environ.get('AWS_SECRET_ACCESS_KEY')]):
        print("Error: AWS credentials not found in environment variables")
        return 1
    
    try:
        # Setup SageMaker
        sagemaker_session, role = setup_sagemaker_session()
        
        # Start finetuning
        print(f"Starting finetuning job for model: {args.base_model}")
        model_data = create_finetune_job(args, sagemaker_session, role)
        
        print(f"Finetuning completed. Model saved to: {model_data}")
        return 0
    except Exception as e:
        print(f"Error during SageMaker finetuning: {e}")
        return 1

if __name__ == "__main__":
    exit(main())