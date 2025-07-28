#!/bin/bash
set -ex

cd /app/llama.cpp
mkdir -p build
cd build

echo "[INFO] CMake build started..."

if cmake .. && cmake --build .; then
    echo "[SUCCESS] Build completed with CURL support."
else
    echo "[WARN] Initial build failed. Retrying with -DLLAMA_CURL=OFF..."
    sleep 1
    cmake .. -DLLAMA_CURL=OFF
    cmake --build .
fi

if [ ! -f bin/quantize ]; then
    echo "[ERROR] quantize binary not found in /app/llama.cpp/build/bin/"
    ls -al bin/
    exit 1
fi

echo "[✓] Build completed. Quantize tool at: /app/llama.cpp/build/bin/quantize"
