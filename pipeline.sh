#!/bin/bash
# Enhanced pipeline script with better logging and error handling
set -e

# Enable debug mode if DEBUG is set
if [ "$DEBUG" = "true" ]; then
  set -x  # Print each command before executing
fi

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

log "Starting Ollama pipeline"
log "Environment variables:"
log "  MODELS_TO_PULL: $MODELS_TO_PULL"
log "  FINETUNE_MODEL: $FINETUNE_MODEL"
log "  TRAINING_DATA: $TRAINING_DATA"
log "  S3_BUCKET: $S3_BUCKET"
log "  AWS_DEFAULT_REGION: $AWS_DEFAULT_REGION"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
  error "AWS CLI is not installed. Installing..."
  pip install awscli
fi

# Wait for Ollama service to be ready
log "Waiting for Ollama service to start..."
for i in {1..30}; do
  if curl -s http://ollama:11434/api/version > /dev/null; then
    log "✅ Ollama service is ready!"
    break
  fi
  log "⏳ Waiting for Ollama... (attempt $i/30)"
  sleep 10
  if [ $i -eq 30 ]; then
    error "Ollama service did not become ready in time"
    exit 1
  fi
done

# Pull models specified in MODELS_TO_PULL
if [ -z "$MODELS_TO_PULL" ]; then
  log "No models specified to pull. Skipping model pull."
else
  log "Models to pull: $MODELS_TO_PULL"
  IFS=' ' read -ra MODELS <<< "$MODELS_TO_PULL"
  for model in "${MODELS[@]}"; do
    log "Pulling model: $model"
    curl -X POST http://ollama:11434/api/pull -d "{\"name\":\"$model\"}"
    if [ $? -ne 0 ]; then
      error "Failed to pull model: $model"
      exit 1
    fi
    log "Model $model pulled successfully"
  done
fi

# Extract GGUF files
log "Extracting GGUF files from Ollama models..."
python extract_gguf.py
if [ $? -ne 0 ]; then
  error "Failed to extract GGUF files"
  exit 1
fi
log "GGUF extraction completed successfully"

# Check if finetuning is enabled
if [ -n "$FINETUNE_MODEL" ] && [ -n "$TRAINING_DATA" ]; then
  log "Finetuning is enabled. Starting finetuning process..."
  
  # Test AWS credentials
  log "Testing AWS credentials..."
  aws sts get-caller-identity
  if [ $? -ne 0 ]; then
    error "AWS credentials issue. Check your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
    exit 1
  fi
  
  # Test S3 access
  log "Testing S3 bucket access..."
  aws s3 ls "$S3_BUCKET" || aws s3 ls "s3://$S3_BUCKET"
  if [ $? -ne 0 ]; then
    error "Cannot access S3 bucket. Check your bucket name and permissions"
    exit 1
  fi
  
  # Check training data exists
  log "Checking training data..."
  aws s3 ls "$TRAINING_DATA" || aws s3 ls "$TRAINING_DATA/"
  if [ $? -ne 0 ]; then
    error "Cannot access training data at $TRAINING_DATA"
    exit 1
  fi
  
  # Run SageMaker finetuning
  log "Starting SageMaker finetuning for $FINETUNE_MODEL"
  python sagemaker_finetune.py \
    --base_model "$FINETUNE_MODEL" \
    --training_data "$TRAINING_DATA" \
    --output_bucket "$S3_BUCKET"
  
  if [ $? -ne 0 ]; then
    error "SageMaker finetuning failed"
    exit 1
  fi
  
  log "Finetuning completed successfully"
  
  # Download finetuned model from S3
  log "Downloading finetuned model from S3..."
  mkdir -p ./finetuned-models
  aws s3 cp "s3://$S3_BUCKET/finetuned-models/" ./finetuned-models/ --recursive
  
  # Quantize the model
  log "Quantizing finetuned model to GGUF format..."
  mkdir -p ./quantized-models
  python scripts/quantize.py \
    --input_model ./finetuned-models/ \
    --output_dir ./quantized-models/ \
    --quantization_level Q4_K_M
  
  if [ $? -ne 0 ]; then
    error "Quantization failed"
    exit 1
  fi
  
  # Create Modelfile
  log "Creating Modelfile for Ollama..."
  mkdir -p ./ollama-models
  python scripts/create_modelfile.py \
    --model_name "finetuned-$(basename $FINETUNE_MODEL)" \
    --gguf_path ./quantized-models/ \
    --output_dir ./ollama-models/
  
  if [ $? -ne 0 ]; then
    error "Modelfile creation failed"
    exit 1
  fi
  
  # Upload results to S3
  log "Uploading results to S3..."
  aws s3 cp ./quantized-models/ "s3://$S3_BUCKET/quantized-models/" --recursive
  aws s3 cp ./ollama-models/ "s3://$S3_BUCKET/ollama-models/" --recursive
  
  log "✅ Finetuning pipeline completed successfully!"
else
  log "Finetuning is not enabled (FINETUNE_MODEL or TRAINING_DATA not set)"
  log "Skipping finetuning process"
fi

log "Pipeline completed at $(date)"