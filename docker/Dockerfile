FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git curl wget zip unzip cmake g++ \
    build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY ../src /app/src
COPY ../scripts /app/scripts
COPY ../requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && pip install -r /app/requirements.txt

RUN git clone https://github.com/ggerganov/llama.cpp.git /app/llama.cpp

# Make build script executable and run it
RUN chmod +x /app/scripts/build_llama.sh && /app/scripts/build_llama.sh

ENTRYPOINT ["/bin/bash"]
