#!/bin/bash
set -ex

cd /app/llama.cpp
mkdir -p build
cd build

echo "[INFO] CMake configuration started..."

# Attempt CMake configuration with CURL support
if cmake .. -DLLAMA_CURL=ON; then
    echo "[SUCCESS] CMake configured with CURL support."
elif cmake ..; then # Fallback to no explicit CURL if first fails
    echo "[SUCCESS] CMake configured without explicit CURL support."
else
    echo "[ERROR] CMake configuration failed."
    exit 1
fi

echo "[INFO] Listing contents of /app/llama.cpp/build before make:"
ls -al .

echo "[INFO] Starting make build..."
# Use make directly to build the project, which should include llama-quantize
if make; then
    echo "[SUCCESS] Make build completed."
else
    echo "[ERROR] Make build failed. Retrying without CURL support if applicable."
    sleep 1
    # If make failed, try re-configuring CMake without CURL and then make again
    cd .. # Go back to llama.cpp root
    rm -rf build # Clean previous build artifacts
    mkdir -p build
    cd build
    if cmake .. -DLLAMA_CURL=OFF && make; then
        echo "[SUCCESS] Make build completed without CURL support."
    else
        echo "[ERROR] Make build failed even after retrying without CURL support."
        exit 1
    fi
fi

echo "[INFO] Listing contents of /app/llama.cpp/build after make:"
ls -al .

echo "[INFO] Listing contents of /app/llama.cpp/build/bin after make:"
ls -al bin/

# Corrected: Check for 'llama-quantize' instead of 'quantize'
if [ ! -f bin/llama-quantize ]; then
    echo "[ERROR] llama-quantize binary not found in /app/llama.cpp/build/bin/ after all build attempts."
    echo "Current directory: $(pwd)"
    echo "Contents of bin/ directory:"
    ls -al bin/ || true # List even if bin doesn't exist or is empty
    exit 1
fi

echo "[✓] Build completed. Quantize tool at: /app/llama.cpp/build/bin/llama-quantize"
