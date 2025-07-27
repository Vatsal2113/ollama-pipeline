# Use compatible PyTorch version (2.1.2) with CUDA 11.8
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Prevent tzdata prompt during install
ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages
RUN apt-get update && apt-get install -y \
    git wget build-essential cmake clang tzdata \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install transformers peft datasets accelerate boto3 sagemaker

# Clone and build llama.cpp using CMake (for GGUF conversion)
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && mkdir build && cd build && \
    cmake .. && cmake --build . --config Release

# Copy your local source code into the container
COPY src /opt/ml/code/

# Set working directory where scripts will be run
WORKDIR /opt/ml/code/
