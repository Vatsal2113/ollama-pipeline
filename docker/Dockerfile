FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (include g++)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    zip \
    unzip \
    cmake \
    build-essential \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY ../src /app/src
COPY ../scripts /app/scripts
COPY ../requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Clone llama.cpp and build GGUF tool
RUN git clone https://github.com/ggerganov/llama.cpp.git /app/llama.cpp && \
    cd /app/llama.cpp && make

# Entry shell
ENTRYPOINT ["/bin/bash"]
