FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Prevent tzdata timezone prompt during build
ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages
RUN apt-get update && apt-get install -y \
    git wget make build-essential cmake clang tzdata \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
RUN pip install --upgrade pip && \
    pip install transformers peft datasets accelerate boto3 sagemaker

# Clone llama.cpp and build quantization tools
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && make

# Copy local source code into container
COPY src /opt/ml/code/

# Set working directory for script execution
WORKDIR /opt/ml/code/
