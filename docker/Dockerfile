FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
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

# Copy your app code
COPY ../src /app/src
COPY ../scripts /app/scripts
COPY ../requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Clone llama.cpp and build using CMake
RUN git clone https://github.com/ggerganov/llama.cpp.git /app/llama.cpp && \
    mkdir -p /app/llama.cpp/build && \
    cd /app/llama.cpp/build && \
    cmake .. && \
    cmake --build . --target quantize

# Set entrypoint
ENTRYPOINT ["/bin/bash"]
