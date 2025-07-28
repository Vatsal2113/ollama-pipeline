#!/bin/bash
set -ex

cd /app/llama.cpp
mkdir -p build
cd build

echo "[INFO] CMake build started..."

# Try building with CURL support
if cmake .. -DLLAMA_CURL=ON && cmake --build .; then
    echo "[SUCCESS] Build completed with CURL support."
else
    echo "[WARN] Initial build failed. Retrying without CURL support..."
    sleep 1
    # Try building without CURL support
    if cmake .. -DLLAMA_CURL=OFF && cmake --build .; then
        echo "[SUCCESS] Build completed without CURL support."
    else
        echo "[ERROR] Failed to build llama.cpp."
        exit 1
    fi
fi

if [ ! -f bin/quantize ]; then
    echo "[ERROR] quantize binary not found in /app/llama.cpp/build/bin/ after all build attempts."
    ls -al bin/
    exit 1
fi

echo "[✓] Build completed. Quantize tool at: /app/llama.cpp/build/bin/quantize"
