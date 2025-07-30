#!/usr/bin/env python3
"""
Script to finetune models using Amazon SageMaker
"""

import os
import argparse
import logging
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_args():
    parser = argparse.ArgumentParser(description="Finetune models using Amazon SageMaker")
    parser.add_argument("--base_model", type=str, required=True, help="Base model to finetune")
    parser.add_argument("--training_data", type=str, required=True, help="S3 URI to training data")
    parser.add_argument("--output_bucket", type=str, required=True, help="S3 bucket for output")
    parser.add_argument("--instance_type", type=str, default="ml.m5.2xlarge", 
                        help="SageMaker instance type (CPU-based by default)")
    return parser.parse_args()

def finetune_model(base_model, training_data, output_bucket, instance_type):
    """Finetune model using Amazon SageMaker"""
    logger.info(f"Setting up SageMaker finetuning for {base_model}")
    
    role = sagemaker.get_execution_role()
    
    # Create a SageMaker session
    session = sagemaker.Session()
    
    # Define output path
    s3_output_location = f"s3://{output_bucket}/finetuned-models/"
    
    # Define hyperparameters for LoRA finetuning
    hyperparameters = {
        'model_name_or_path': base_model,
        'output_dir': '/opt/ml/model',
        'num_train_epochs': 3,
        'per_device_train_batch_size': 4,
        'gradient_accumulation_steps': 2,
        'learning_rate': 2e-4,
        'warmup_steps': 10,
        
        # LoRA specific parameters
        'use_lora': True,
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        
        # Quantization for memory efficiency
        'load_in_8bit': True,
        
        # Use Flash Attention when available
        'use_flash_attention': False,  # Set to False for CPU
    }
    
    # Create HuggingFace estimator
    huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts/training_scripts',
        instance_type=instance_type,
        instance_count=1,
        role=role,
        transformers_version='4.28.1',
        pytorch_version='2.0.0',
        py_version='py39',
        hyperparameters=hyperparameters,
        output_path=s3_output_location,
    )
    
    # Start training
    logger.info("Starting SageMaker training job...")
    huggingface_estimator.fit({'train': training_data})
    
    logger.info(f"Training complete! Model saved to {s3_output_location}")
    return s3_output_location

def main():
    args = setup_args()
    finetune_model(args.base_model, args.training_data, args.output_bucket, args.instance_type)

if __name__ == "__main__":
    main()