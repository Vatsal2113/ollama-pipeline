FROM ubuntu:22.04

# Prevent prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    python3 \
    python3-pip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

# Copy scripts
COPY pipeline.sh extract_gguf.py sagemaker_finetune.py ./
COPY scripts ./scripts/

# Make scripts executable
RUN chmod +x pipeline.sh extract_gguf.py sagemaker_finetune.py

# Environment variables
ENV S3_BUCKET="ollama-lora-pipeline"
ENV AWS_REGION="us-east-2"
ENV MODELS_TO_PULL="llama2 mistral"
ENV OUTPUT_DIR="./models"
# Set these to enable finetuning
# ENV FINETUNE_MODEL="mistralai/Mistral-7B-v0.1"
# ENV TRAINING_DATA="s3://ollama-lora-pipeline/training-data/"

# Run the pipeline when container starts
CMD ["./pipeline.sh"]