FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (including libcurl for optional HTTP)
RUN apt-get update && apt-get install -y \
    git curl wget zip unzip cmake g++ \
    build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy your source, scripts, and requirements
# SageMaker expects the training script at /opt/ml/code
COPY src /opt/ml/code/src
COPY scripts /opt/ml/code/scripts
COPY requirements.txt /opt/ml/code/requirements.txt

# Install Python dependencies
# Unsloth will try to install the correct torch version (CPU or CUDA) based on environment
RUN pip install --upgrade pip && pip install -r /opt/ml/code/requirements.txt

# Set the entrypoint for SageMaker training job
# SageMaker will execute 'python3 /opt/ml/code/src/finetune_lora.py'
ENTRYPOINT ["/bin/bash"]
