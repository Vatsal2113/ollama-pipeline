#!/bin/bash
set -e

echo "Starting Ollama pipeline at $(date)"

# Wait for Ollama service to be ready
max_retries=30
count=0
while [ $count -lt $max_retries ]
do
  if curl -s http://ollama:11434/api/version > /dev/null; then
    echo "Ollama service is ready!"
    break
  fi
  echo "Waiting for Ollama service to be ready... ($((count+1))/$max_retries)"
  sleep 5
  count=$((count+1))
done

if [ $count -eq $max_retries ]; then
  echo "Error: Ollama service is not running. Please start Ollama first."
  exit 1
fi

# Continue with rest of your script
# Pull models
# Extract GGUFs
# Run finetuning if enabled
# etc.