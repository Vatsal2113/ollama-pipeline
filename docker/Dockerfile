FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages (CMake + build tools)
RUN apt-get update && apt-get install -y \
    git wget build-essential cmake clang tzdata \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install transformers peft datasets accelerate boto3 sagemaker

# Clone and build llama.cpp using CMake
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && mkdir build && cd build && \
    cmake .. && cmake --build . --config Release

# Copy your code
COPY src /opt/ml/code/
WORKDIR /opt/ml/code/
