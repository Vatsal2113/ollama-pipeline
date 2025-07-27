#!/bin/bash
set -e

MODEL_NAME="$1"
JSONL_PATH="$2"

echo "[*] Fine-tuning model: $MODEL_NAME"
echo "[*] Dataset: $JSONL_PATH"

python3 finetune_lora.py "$MODEL_NAME" "$JSONL_PATH"
python3 convert_to_gguf.py "$MODEL_NAME"

echo "[*] Uploading to S3..."
aws s3 cp "./artifacts/${MODEL_NAME//\//_}/gguf/model.gguf" \
  "s3://ollama-lora-pipeline/gguf/${MODEL_NAME//\//_}.gguf"

echo "[*] Pipeline completed successfully!"
