FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (including libcurl for optional HTTP)
RUN apt-get update && apt-get install -y \
    git curl wget zip unzip cmake g++ \
    build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy your source, scripts, and requirements
COPY ../src /app/src
COPY ../scripts /app/scripts
COPY ../requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Clone llama.cpp before running build script!
RUN git clone https://github.com/ggerganov/llama.cpp.git /app/llama.cpp

# Copy and run the build script for llama.cpp
COPY ../scripts/build_llama.sh /app/scripts/build_llama.sh
RUN chmod +x /app/scripts/build_llama.sh && bash -x /app/scripts/build_llama.sh

ENTRYPOINT ["/bin/bash"]
