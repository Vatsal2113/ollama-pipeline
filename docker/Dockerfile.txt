FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y git wget make && \
    pip install --upgrade pip && \
    pip install transformers peft datasets accelerate boto3 sagemaker

# Clone llama.cpp for quantization (optional: pin commit)
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && make

COPY src /opt/ml/code/
WORKDIR /opt/ml/code/
