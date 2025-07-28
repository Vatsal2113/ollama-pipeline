#!/bin/bash
set -e

cd /app/llama.cpp
mkdir -p build
cd build

echo "[INFO] Trying to build with CURL support..."
if cmake .. && cmake --build . --target ALL; then
    echo "[SUCCESS] Built with CURL"
else
    echo "[WARN] Failed with CURL, retrying with -DLLAMA_CURL=OFF..."
    cmake .. -DLLAMA_CURL=OFF
    cmake --build . --target ALL
fi
