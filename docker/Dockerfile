FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install required system packages
RUN apt-get update && apt-get install -y \
    git wget make build-essential cmake clang \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install transformers peft datasets accelerate boto3 sagemaker

# Clone and build llama.cpp for GGUF conversion
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && make

# Copy source code into the container
COPY src /opt/ml/code/

# Set working directory
WORKDIR /opt/ml/code/
