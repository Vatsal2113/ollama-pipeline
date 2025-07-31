#!/usr/bin/env python3
import os
import time
import random
import string
import argparse
import logging
import boto3
import sagemaker
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
        help="Base model to fine-tune (e.g., mistralai/Mistral-7B-v0-1)"
    )
    
    parser.add_argument(
        "--training-data",
        required=True,
        help="S3 URI of training data (e.g., s3://bucket/data/)"
    )
    
    parser.add_argument(
        "--output-bucket",
        required=True,
        help="S3 bucket for output model"
    )
    
    parser.add_argument(
        "--instance-type",
        default=os.environ.get("TRAINING_INSTANCE_TYPE", "ml.p3.2xlarge"),
        help="SageMaker training instance type"
    )
    
    parser.add_argument(
        "--max-runtime",
        type=int,
        default=86400,  # 24 hours
        help="Maximum runtime in seconds"
    )
    
    args = parser.parse_args()
    
    # Log the arguments
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Training data: {args.training_data}")
    logger.info(f"Output bucket: {args.output_bucket}")
    logger.info(f"Instance type: {args.instance_type}")
    
    return args

def finetune_model(base_model, training_data, output_bucket, instance_type, max_runtime=86400):
    """Fine-tune a model using SageMaker."""
    logger.info(f"Setting up SageMaker finetuning for {base_model}")
    
    # Get the AWS account ID
    account_id = boto3.client('sts').get_caller_identity().get('Account')
    logger.info(f"AWS Account ID: {account_id}")
    
    # Set up SageMaker session
    sagemaker_session = sagemaker.Session(default_bucket=output_bucket)
    logger.info(f"SageMaker session created successfully with bucket: {output_bucket}")
    
    # Get the execution role from environment or use default
    role = os.environ.get("SAGEMAKER_ROLE_ARN")
    logger.info(f"Using SageMaker execution role: {role}")
    
    # Configure hyperparameters
    hyperparameters = {
        "model_name_or_path": base_model,
        "output_dir": "/opt/ml/model",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-5,
        "fp16": True,
        "save_strategy": "steps",
        "save_steps": 500,
    }
    
    # Set up the Hugging Face estimator
    huggingface_estimator = HuggingFace(
        entry_point="train.py",
        source_dir="./scripts/training_scripts",
        instance_type=instance_type,
        instance_count=1,
        role=role,
        transformers_version="4.26",
        pytorch_version="1.13",
        py_version="py39",
        hyperparameters=hyperparameters,
        max_run=max_runtime,
        dependencies=["./scripts/training_scripts/requirements.txt"]
    )
    logger.info("HuggingFace estimator created successfully")
    
    # Create training job inputs
    inputs = {"training": training_data}
    
    # Start the training job
    try:
        # Generate a unique job name
        timestamp = int(time.time())
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        job_name = f"{base_model.replace('/', '-').lower()}-{timestamp}-{random_suffix}"
        
        logger.info(f"Creating training-job with name: {job_name}")
        huggingface_estimator.fit(inputs=inputs, job_name=job_name)
        logger.info(f"Training job {job_name} started successfully")
        
        try:
            # Wait for job to complete
            huggingface_estimator.wait(logs="All")
            logger.info(f"Training job {job_name} completed")
            
            # Get model artifacts
            model_artifacts = huggingface_estimator.model_data
            logger.info(f"Model artifacts stored at: {model_artifacts}")
            return model_artifacts, job_name
        
        except Exception as e:
            logger.warning(f"Error waiting for job or accessing logs: {str(e)}")
            logger.info(f"Training job {job_name} is running in the background")
            logger.info(f"Check AWS SageMaker console for status and logs")
            return None, job_name
            
    except Exception as e:
        logger.error(f"Error starting training job: {str(e)}")
        raise e

def main():
    args = setup_args()
    try:
        model_artifacts, job_name = finetune_model(
            args.base_model, 
            args.training_data, 
            args.output_bucket, 
            args.instance_type,
            args.max_runtime
        )
        
        if model_artifacts:
            logger.info(f"Training completed successfully")
            logger.info(f"Model artifacts: {model_artifacts}")
        else:
            logger.info(f"Training job {job_name} started but couldn't monitor progress")
            logger.info("Please check the AWS SageMaker console for the job status")
            
    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()