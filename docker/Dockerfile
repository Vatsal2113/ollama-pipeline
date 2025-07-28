FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including libcurl
RUN apt-get update && apt-get install -y \
    git curl wget zip unzip cmake g++ \
    build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY ../src /app/src
COPY ../scripts /app/scripts
COPY ../requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Clone llama.cpp and build using CMake
RUN git clone https://github.com/ggerganov/llama.cpp.git /app/llama.cpp && \
    mkdir -p /app/llama.cpp/build

# Try building with CURL; if it fails, retry with -DLLAMA_CURL=OFF
RUN cd /app/llama.cpp/build && \
    (cmake .. && cmake --build . --target quantize) || \
    (echo "[WARN] CURL build failed, retrying with -DLLAMA_CURL=OFF..." && \
     cmake .. -DLLAMA_CURL=OFF && \
     cmake --build . --target quantize)

ENTRYPOINT ["/bin/bash"]
