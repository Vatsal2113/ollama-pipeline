#!/bin/bash

# Get the latest training job name
JOB_NAME=$(aws sagemaker list-training-jobs --sort-by "CreationTime" --sort-order "Descending" --max-items 1 --query "TrainingJobSummaries[0].TrainingJobName" --output text)

echo "Latest training job: $JOB_NAME"

# Set variables
BASE_MODEL="mistralai/Mistral-7B-v0-1"
MODEL_NAME="mistral-custom"
FINE_TUNED_DIR="./fine-tuned-model"
MERGED_DIR="./merged-model"
GGUF_PATH="./models/${MODEL_NAME}.gguf"

# Download fine-tuned model from SageMaker
echo "Downloading fine-tuned model..."
python scripts/download_model.py --job-name $JOB_NAME --output-dir $FINE_TUNED_DIR

# Merge LoRA weights with base model
echo "Merging LoRA weights with base model..."
python scripts/merge_lora.py --base-model $BASE_MODEL --lora-model $FINE_TUNED_DIR --output-dir $MERGED_DIR

# Convert to GGUF format
echo "Converting model to GGUF format..."
python scripts/convert_to_gguf.py --input-dir $MERGED_DIR --output-file $GGUF_PATH --outtype q4_k_m

# Create Ollama Modelfile
echo "Creating Ollama Modelfile..."
python scripts/create_modelfile.py --gguf-path $GGUF_PATH --model-name $MODEL_NAME

echo "Pipeline complete! Your model is ready for Ollama."
echo "To import into Ollama, run: ollama create $MODEL_NAME -f Modelfile"
echo "To run your model: ollama run $MODEL_NAME"