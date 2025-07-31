#!/usr/bin/env python3
import os
import argparse
import logging
import boto3
import sagemaker
import time
import string
import random
from sagemaker.huggingface import HuggingFace

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("sagemaker_finetune")

def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a model with SageMaker")
    
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model to fine-tune"
    )
    
    parser.add_argument(
        "--training-data",
        required=True,
        help="S3 URI of training data"
    )
    
    parser.add_argument(
        "--output-bucket",
        required=True,
        help="S3 bucket for output model"
    )
    
    parser.add_argument(
        "--instance-type",
        default="ml.g4dn.xlarge",
        help="SageMaker training instance type"
    )
    
    return parser.parse_args()

def main():
    args = setup_args()
    try:
        logger.info(f"Setting up SageMaker finetuning for {args.base_model}")
        
        # Set up SageMaker session
        sagemaker_session = sagemaker.Session()
        role = os.environ.get("SAGEMAKER_ROLE_ARN")
        logger.info(f"Using role: {role}")
        
        # Define hyperparameters for fine-tuning
        hyperparameters = {
            'model_name_or_path': args.base_model,
            'output_dir': '/opt/ml/model',
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 4,
            'learning_rate': 2e-4,
            'num_train_epochs': 1,
            'use_lora': 'False'  # Disabled for now
        }
        
        # Generate a simple job name
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
        job_name = f"train-{suffix}"
        
        # Use Hugging Face estimator with compatible versions for Python 3.11
        huggingface_estimator = HuggingFace(
            entry_point='train_lora.py',
            source_dir='./scripts/training_scripts/lora_trainer',
            instance_type=args.instance_type,
            instance_count=1,
            role=role,
            transformers_version='4.49.0',  # Compatible with Python 3.11
            pytorch_version='2.5.1',        # Compatible with Python 3.11
            py_version='py311',             # Using Python 3.11 as requested
            hyperparameters=hyperparameters,
            debugger_hook_config=False
        )
        
        # Create training job inputs
        inputs = {'training': args.training_data}
        
        # Start the training job
        logger.info(f"Starting training job: {job_name}")
        huggingface_estimator.fit(inputs=inputs, job_name=job_name)
        logger.info(f"Training job {job_name} completed")
        
        # Get the model artifacts
        model_data = huggingface_estimator.model_data
        logger.info(f"Model artifacts: {model_data}")
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()