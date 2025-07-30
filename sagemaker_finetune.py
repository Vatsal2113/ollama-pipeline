#!/usr/bin/env python3
import os
import sys
import boto3
import logging
import argparse
import sagemaker
from sagemaker.huggingface import HuggingFace

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("sagemaker_finetune")

def setup_args():
    """Setup and parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune a model using SageMaker')
    
    parser.add_argument('--base-model', type=str, required=True,
                        help='Base model name or HuggingFace model ID')
    
    parser.add_argument('--training-data', type=str, required=True,
                        help='S3 URI for training data')
    
    parser.add_argument('--output-bucket', type=str, required=True,
                        help='S3 bucket for output artifacts')
    
    parser.add_argument('--instance-type', type=str, default='ml.g5.2xlarge',
                        help='SageMaker instance type for training')
    
    parser.add_argument('--max-train-samples', type=int, default=1000,
                        help='Maximum number of training samples')
    
    parser.add_argument('--num-train-epochs', type=int, default=3,
                        help='Number of training epochs')
    
    parser.add_argument('--per-device-train-batch-size', type=int, default=8,
                        help='Per device training batch size')
    
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                        help='Number of gradient accumulation steps')
    
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                        help='Learning rate')
    
    parser.add_argument('--output-model-name', type=str, default='finetuned-model',
                        help='Name for the output model')
    
    args = parser.parse_args()
    
    # Log arguments
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Training data: {args.training_data}")
    logger.info(f"Output bucket: {args.output_bucket}")
    logger.info(f"Instance type: {args.instance_type}")
    
    return args

def finetune_model(base_model, training_data, output_bucket, instance_type='ml.g5.2xlarge'):
    """Finetune model using Amazon SageMaker"""
    logger.info(f"Setting up SageMaker finetuning for {base_model}")
    
    # Verify AWS session can be created
    try:
        boto_session = boto3.Session()
        account_id = boto_session.client('sts').get_caller_identity()['Account']
        logger.info(f"AWS Account ID: {account_id}")
    except Exception as e:
        logger.error(f"Error creating AWS session: {str(e)}")
        raise
        
    # Create a SageMaker session with the specified bucket
    try:
        # Use the provided bucket instead of default
        session = sagemaker.Session(boto_session=boto_session, default_bucket=output_bucket)
        logger.info(f"SageMaker session created successfully with bucket: {output_bucket}")
    except Exception as e:
        logger.error(f"Error creating SageMaker session: {str(e)}")
        raise
    
    # Get SageMaker execution role from environment variable
    try:
        role = os.environ.get("SAGEMAKER_ROLE_ARN")
        if not role:
            logger.error("SAGEMAKER_ROLE_ARN environment variable not set")
            raise ValueError("SAGEMAKER_ROLE_ARN environment variable is required")
        logger.info(f"Using SageMaker execution role: {role}")
    except Exception as e:
        logger.error(f"Error getting SageMaker execution role: {str(e)}")
        raise
    
    # Configure hyperparameters
    hyperparameters = {
        'model_name_or_path': base_model,
        'output_dir': '/opt/ml/model',
        'max_train_samples': 1000,
        'num_train_epochs': 3,
        'per_device_train_batch_size': 8,
        'gradient_accumulation_steps': 4,
        'learning_rate': 2e-4,
        'fp16': True,
    }
    
    # Set environment variables for the training container
    environment = {
        'TRAINING_DATA_PATH': training_data,
        'USE_LORA': 'true',
        'LORA_R': '16',
        'LORA_ALPHA': '32',
        'LORA_DROPOUT': '0.05',
    }
    
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
            environment=environment,
            max_run=24 * 60 * 60,  # 24 hours
            sagemaker_session=session  # Pass the session with custom bucket
        )
        logger.info("HuggingFace estimator created successfully")
    except Exception as e:
        logger.error(f"Error creating HuggingFace estimator: {str(e)}")
        raise
    
    # Start the training job
    try:
        job_name = f"{base_model.replace('/', '-')}-finetuned"
        huggingface_estimator.fit(job_name=job_name)
        logger.info(f"Training job {job_name} started successfully")
    except Exception as e:
        logger.error(f"Error starting training job: {str(e)}")
        raise
    
    # Get the training job name
    training_job_name = huggingface_estimator.latest_training_job.job_name
    logger.info(f"Training job name: {training_job_name}")
    
    # Get the model artifacts from S3
    model_artifacts = huggingface_estimator.model_data
    logger.info(f"Model artifacts: {model_artifacts}")
    
    # Convert the finetuned model to GGUF format using a separate process
    output_file = f"{output_bucket}/ollama-models/finetuned-{base_model.split('/')[-1]}.gguf"
    logger.info(f"Output model will be saved to: {output_file}")
    
    return training_job_name, model_artifacts, output_file

def main():
    args = setup_args()
    finetune_model(args.base_model, args.training_data, args.output_bucket, args.instance_type)

if __name__ == "__main__":
    main()