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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    
    # Verify AWS session can be created
    try:
        boto_session = boto3.Session()
        logger.info(f"AWS Account ID: {boto_session.client('sts').get_caller_identity()['Account']}")
    except Exception as e:
        logger.error(f"Error creating AWS session: {str(e)}")
        raise
        
    # Create a SageMaker session
    try:
        session = sagemaker.Session(boto_session=boto_session)
        logger.info(f"SageMaker session created successfully")
    except Exception as e:
        logger.error(f"Error creating SageMaker session: {str(e)}")
        raise
    
    # Get SageMaker execution role
    try:
        role = sagemaker.get_execution_role()
        logger.info(f"Using SageMaker execution role: {role}")
    except Exception as e:
        logger.warning(f"Error getting SageMaker execution role: {str(e)}")
        logger.warning("Using empty role string - AWS will use the role associated with your AWS credentials")
        role = ""
    
    # Define output path
    s3_output_location = f"s3://{output_bucket}/finetuned-models/"
    logger.info(f"Model output will be saved to: {s3_output_location}")
    
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
        
        # Use Flash Attention when available (not for CPU)
        'use_flash_attention': False,
    }
    
    logger.info("Creating HuggingFace estimator...")
    
    # Create HuggingFace estimator
    try:
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
        logger.info("HuggingFace estimator created successfully")
    except Exception as e:
        logger.error(f"Error creating HuggingFace estimator: {str(e)}")
        raise
    
    # Start training
    logger.info("Starting SageMaker training job...")
    try:
        huggingface_estimator.fit({'train': training_data})
        logger.info(f"Training complete! Model saved to {s3_output_location}")
    except Exception as e:
        logger.error(f"Error during SageMaker training: {str(e)}")
        raise
        
    return s3_output_location

def main():
    args = setup_args()
    finetune_model(args.base_model, args.training_data, args.output_bucket, args.instance_type)

if __name__ == "__main__":
    main()