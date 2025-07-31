#!/bin/bash
set -e

# Enable debug mode if requested
if [ "$DEBUG" = "true" ]; then
  set -x
fi

echo "Ollama Pipeline starting at $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo "User: Vatsal2113"

# Check for required environment variables
for var in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_DEFAULT_REGION S3_BUCKET SAGEMAKER_ROLE_ARN; do
  if [ -z "${!var}" ]; then
    echo "Error: $var environment variable is not set"
    exit 1
  fi
done

# Verify AWS credentials work
echo "Verifying AWS credentials..."
aws sts get-caller-identity || {
  echo "Error: Failed to authenticate with AWS. Check your credentials."
  exit 1
}

# Pull required models for Ollama (if MODELS_TO_PULL is set)
if [ -n "$MODELS_TO_PULL" ]; then
  echo "Pulling Ollama models: $MODELS_TO_PULL"
  # Check if curl is installed
  if ! command -v curl &> /dev/null; then
    echo "Warning: curl is not installed. Installing curl..."
    apt-get update && apt-get install -y curl
  fi
  
  for model in $MODELS_TO_PULL; do
    echo "Pulling $model..."
    curl -X POST http://ollama:11434/api/pull -d "{\"name\":\"$model\"}" || {
      echo "Warning: Failed to pull model $model. Continuing..."
    }
  done
fi

# Run fine-tuning with SageMaker
if [ -n "$FINETUNE_MODEL" ]; then
  echo "Setting up fine-tuning for $FINETUNE_MODEL..."
  
  # Create sample training data if needed
  if [ "$CREATE_SAMPLE_DATA" = "true" ]; then
    echo "Creating sample training data..."
    mkdir -p /tmp/training_data
    echo '{"prompt": "What is machine learning?", "completion": "Machine learning is a branch of AI that allows systems to learn and improve from experience."}' > /tmp/training_data/sample.jsonl
    aws s3 cp /tmp/training_data/ s3://$S3_BUCKET/training-data/ --recursive
  fi
  
  # Run the fine-tuning script
  echo "Starting SageMaker fine-tuning..."
  python3 sagemaker_finetune.py \
    --base-model "$FINETUNE_MODEL" \
    --training-data "$TRAINING_DATA" \
    --output-bucket "$S3_BUCKET" \
    --instance-type "ml.m5.4xlarge"
  
  echo "Fine-tuning complete!"
fi

echo "Pipeline completed at $(date -u +"%Y-%m-%d %H:%M:%S UTC")"