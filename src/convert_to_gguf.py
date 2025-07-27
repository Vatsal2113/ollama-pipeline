import sys
import os
import subprocess

model_id = sys.argv[1]
model_dir = f"./artifacts/{model_id.replace('/', '_')}/finetuned"
gguf_output_dir = f"./artifacts/{model_id.replace('/', '_')}/gguf"

os.makedirs(gguf_output_dir, exist_ok=True)

cmd = [
    "./llama.cpp/quantize",
    os.path.join(model_dir, "pytorch_model.bin"),
    os.path.join(gguf_output_dir, "model.gguf"),
    "q4_0"
]

print(f"[INFO] Running quantization: {' '.join(cmd)}")

try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    print(f"[ERROR] GGUF conversion failed: {e}")
    sys.exit(1)
