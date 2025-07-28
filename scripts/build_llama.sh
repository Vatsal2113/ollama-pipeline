#!/bin/bash
set -ex

cd /app/llama.cpp
mkdir -p build
cd build

echo "[INFO] CMake build started..."

# Try building with CURL support and explicitly target quantize
if cmake .. -DLLAMA_CURL=ON && cmake --build . --target quantize; then
    echo "[SUCCESS] Build completed with CURL support, quantize target built."
elif cmake .. && cmake --build . --target quantize; then
    echo "[SUCCESS] Build completed without explicit CURL, quantize target built."
else
    echo "[WARN] Initial build of quantize failed. Retrying without CURL support..."
    sleep 1
    # Try building without CURL support and explicitly target quantize
    if cmake .. -DLLAMA_CURL=OFF && cmake --build . --target quantize; then
        echo "[SUCCESS] Quantize target built without CURL support."
    else
        echo "[ERROR] Failed to build quantize target even without CURL support."
        exit 1
    fi
fi

if [ ! -f bin/quantize ]; then
    echo "[ERROR] quantize binary not found in /app/llama.cpp/build/bin/ after all build attempts."
    ls -al bin/
    exit 1
fi

echo "[✓] Build completed. Quantize tool at: /app/llama.cpp/build/bin/quantize"
