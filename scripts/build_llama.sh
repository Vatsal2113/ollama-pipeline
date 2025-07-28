#!/bin/bash
set -e

cd /app/llama.cpp
mkdir -p build
cd build

echo "[INFO] Attempting to build llama.cpp tools (e.g., quantize)..."

# Try with CURL support
if cmake .. && cmake --build .; then
    echo "[SUCCESS] Built with default settings"
else
    echo "[WARN] Failed. Retrying with -DLLAMA_CURL=OFF..."
    cmake .. -DLLAMA_CURL=OFF
    cmake --build .
fi

# Confirm quantize binary exists
if [ ! -f bin/quantize ]; then
    echo "[ERROR] quantize tool not found after build."
    exit 1
fi

echo "[✓] quantize built successfully at: /app/llama.cpp/build/bin/quantize"
