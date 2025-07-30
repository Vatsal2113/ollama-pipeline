#!/bin/bash
set -e

echo "Starting Ollama pipeline at $(date)"

# Wait for Ollama service to be ready
echo "Waiting for Ollama service to start..."
for i in {1..30}; do
  if curl -s http://ollama:11434/api/version > /dev/null; then
    echo "✅ Ollama service is ready!"
    break
  fi
  echo "⏳ Waiting for Ollama... (attempt $i/30)"
  sleep 10
  if [ $i -eq 30 ]; then
    echo "❌ Ollama service did not become ready in time"
    exit 1
  fi
done

# Pull models specified in MODELS_TO_PULL
IFS=' ' read -ra MODELS <<< "$MODELS_TO_PULL"
for model in "${MODELS[@]}"; do
  echo "Pulling model: $model"
  curl -X POST http://ollama:11434/api/pull -d "{\"name\":\"$model\"}"
  echo "Model $model pulled successfully"
done

# Extract GGUF files if needed
echo "Extracting GGUF files from Ollama models..."
python extract_gguf.py

# Check if finetuning is enabled
if [ -n "$FINETUNE_MODEL" ] && [ -n "$TRAINING_DATA" ]; then
  echo "Finetuning is enabled. Starting finetuning process..."
  
  # Run SageMaker finetuning
  echo "Starting SageMaker finetuning for $FINETUNE_MODEL"
  python sagemaker_finetune.py \
    --base_model "$FINETUNE_MODEL" \
    --training_data "$TRAINING_DATA" \
    --output_bucket "$S3_BUCKET"
  
  # Download finetuned model from S3
  echo "Downloading finetuned model from S3..."
  aws s3 cp s3://$S3_BUCKET/finetuned-models/ ./finetuned-models/ --recursive
  
  # Quantize the finetuned model to GGUF format
  echo "Quantizing finetuned model to GGUF format..."
  python scripts/quantize.py \
    --input_model ./finetuned-models/ \
    --output_dir ./quantized-models/ \
    --quantization_level Q4_K_M
  
  # Create Modelfile for Ollama
  echo "Creating Modelfile for Ollama..."
  python scripts/create_modelfile.py \
    --model_name "finetuned-$(basename $FINETUNE_MODEL)" \
    --gguf_path ./quantized-models/ \
    --output_dir ./ollama-models/
  
  # Upload GGUF and Modelfile to S3
  echo "Uploading quantized model and Modelfile to S3..."
  aws s3 cp ./quantized-models/ s3://$S3_BUCKET/quantized-models/ --recursive
  aws s3 cp ./ollama-models/ s3://$S3_BUCKET/ollama-models/ --recursive
  
  echo "✅ Finetuning, quantization, and model creation complete!"
else
  echo "Finetuning is not enabled. Skipping finetuning process."
fi

echo "Pipeline completed successfully!"