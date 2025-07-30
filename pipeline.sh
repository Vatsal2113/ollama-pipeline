#!/bin/bash
set -e

# Configuration
S3_BUCKET=${S3_BUCKET:-"ollama-lora-pipeline"}
MODELS_TO_PULL=${MODELS_TO_PULL:-"llama2 mistral"}
OUTPUT_DIR=${OUTPUT_DIR:-"./models"}
FINETUNE_MODEL=${FINETUNE_MODEL:-""}  # Set to model name to enable finetuning
TRAINING_DATA=${TRAINING_DATA:-""}    # S3 path to training data

echo "Starting Ollama pipeline at $(date)"

# Check if ollama is running
if ! curl -s http://localhost:11434/api/version > /dev/null; then
    echo "Error: Ollama service is not running. Please start Ollama first."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/modelfiles"

# Check AWS configuration
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "Error: AWS credentials not configured correctly"
    exit 1
fi

# Pull models
for model in $MODELS_TO_PULL; do
    echo "Pulling model: $model"
    ollama pull $model || {
        echo "Error: Failed to pull model $model"
        exit 1
    }
    
    # Extract modelfile
    echo "Extracting modelfile for $model"
    ollama show $model --modelfile > "$OUTPUT_DIR/modelfiles/$model.modelfile" || {
        echo "Error: Failed to extract modelfile for $model"
    }
done

# Extract GGUF files
echo "Extracting GGUF files"
python3 extract_gguf.py --output "$OUTPUT_DIR" || {
    echo "Error: Failed to extract GGUF files"
    exit 1
}

# Upload to S3
echo "Uploading to S3"
aws s3 sync "$OUTPUT_DIR" "s3://$S3_BUCKET/" || {
    echo "Error: Failed to upload to S3"
    exit 1
}

# Run finetuning if requested
if [ -n "$FINETUNE_MODEL" ] && [ -n "$TRAINING_DATA" ]; then
    echo "Starting finetuning for model: $FINETUNE_MODEL"
    python3 sagemaker_finetune.py \
        --s3-bucket "$S3_BUCKET" \
        --base-model "$FINETUNE_MODEL" \
        --training-data "$TRAINING_DATA" \
        --output-path "s3://$S3_BUCKET/finetuned-models/" || {
        echo "Error: Finetuning failed"
        exit 1
    }
fi

echo "Pipeline completed successfully at $(date)"