import sys
import os
import subprocess

model_id = sys.argv[1]

# Define paths
model_dir = f"./artifacts/{model_id.replace('/', '_')}/finetuned"
gguf_output_dir = f"./artifacts/{model_id.replace('/', '_')}/gguf"
quantize_path = "/app/llama.cpp/build/bin/quantize"

# Create output directory if needed
os.makedirs(gguf_output_dir, exist_ok=True)

# Build full command
input_model_path = os.path.join(model_dir, "pytorch_model.bin")
output_model_path = os.path.join(gguf_output_dir, "model.gguf")

cmd = [quantize_path, input_model_path, output_model_path, "q4_0"]

print("[INFO] Running GGUF quantization...")
print("       Command:", " ".join(cmd))

# Run quantization
try:
    subprocess.run(cmd, check=True)
    print(f"[SUCCESS] GGUF model saved to: {output_model_path}")
except subprocess.CalledProcessError as e:
    print(f"[ERROR] GGUF conversion failed: {e}")
    sys.exit(1)
