# Use compatible PyTorch image with CUDA 11.8 and PyTorch >= 2.1
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Avoid interactive prompts from tzdata
ENV DEBIAN_FRONTEND=noninteractive

# Install system-level build tools and dependencies
RUN apt-get update && apt-get install -y \
    git wget build-essential cmake clang tzdata \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages, including tiktoken for OpenLLaMA tokenizers
RUN pip install --upgrade pip && \
    pip install transformers peft datasets accelerate boto3 sagemaker tiktoken blobfile

# Clone and build llama.cpp using CMake (for GGUF conversion)
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && mkdir build && cd build && \
    cmake .. && cmake --build . --config Release

# Copy source code into container
COPY src /opt/ml/code/

# Set working directory for script execution
WORKDIR /opt/ml/code/
