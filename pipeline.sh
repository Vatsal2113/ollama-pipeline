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

# Rest of your pipeline code goes here
# Pull models specified in MODELS_TO_PULL
for model in ${MODELS_TO_PULL}; do
  echo "Pulling model: $model"
  curl -X POST http://ollama:11434/api/pull -d "{\"name\":\"$model\"}"
  echo "Model $model pulled successfully"
done

# Continue with the rest of your pipeline...