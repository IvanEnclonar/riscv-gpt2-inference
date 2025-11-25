"""
ablate.py — Run prompt ablations and collect performance metrics.
"""

import sys
import subprocess
import os
from transformers import GPT2Tokenizer

# --- CONFIGURATION ---
SOURCE_FILE = "inference.c"
BINARY_NAME = "gpt2_riscv"
OUTPUT_FILE = "performance.txt"
COMPILER = "riscv64-linux-gnu-gcc"
EMULATOR = "qemu-riscv64"

CFLAGS = ["-Ofast", "-march=rv64gcv", "-mabi=lp64d", "-lm", "-static"]

ABLATIONS = [
    {"name": "Scalar Baseline",           "matmul": 0, "ln": 0, "add": 0},
    {"name": "RVV MatMul Only",           "matmul": 1, "ln": 0, "add": 0},
    {"name": "RVV LayerNorm Only",        "matmul": 0, "ln": 1, "add": 0},
    {"name": "RVV MatMul + LN",           "matmul": 1, "ln": 1, "add": 0},
    {"name": "RVV Add Only",              "matmul": 0, "ln": 0, "add": 1},
    {"name": "RVV MatMul + Add",          "matmul": 1, "ln": 0, "add": 1},
    {"name": "RVV LN + Add",              "matmul": 0, "ln": 1, "add": 1},
    {"name": "Fully Vectorized (RVV All)", "matmul": 1, "ln": 1, "add": 1},
]

def compile_binary(matmul, ln, add):
    print(f"  [Compiling] MatMul={matmul}, LN={ln}, Add={add}...")
    defines = [f"-DENABLE_RVV_MATMUL={matmul}", f"-DENABLE_RVV_LAYERNORM={ln}", f"-DENABLE_RVV_ADD={add}"]
    cmd = [COMPILER, SOURCE_FILE, "-o", BINARY_NAME] + CFLAGS + defines
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("  [Error] Compilation failed.")
        sys.exit(1)

def run_simulation(num_tokens, prompt_tokens_str):
    # To count real memory accesses, add: 
    # -plugin /path/to/libmem.so,inline=true,callback=true -d plugin
    cmd = [EMULATOR, f"./{BINARY_NAME}", str(num_tokens), prompt_tokens_str]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout.strip().split('\n')
    except subprocess.CalledProcessError as e:
        print(f"  [Error] Simulation failed: {e}")
        return None

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 ablate.py <num_output_tokens> \"<prompt>\"")
        sys.exit(1)

    output_token_count = int(sys.argv[1])
    raw_prompt = sys.argv[2]
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode(raw_prompt)
    prompt_tokens_str = ",".join(map(str, input_ids))

    with open(OUTPUT_FILE, "w") as f:
        f.write(f"Ablation Report\nPrompt: {raw_prompt}\nGenerating: {output_token_count}\n" + "="*60 + "\n\n")

    for idx, config in enumerate(ABLATIONS):
        print(f"\nRunning {idx+1}/{len(ABLATIONS)}: {config['name']}")
        compile_binary(config["matmul"], config["ln"], config["add"])
        
        output_lines = run_simulation(output_token_count, prompt_tokens_str)
        if not output_lines or len(output_lines) < 6:
            print("  [Error] Invalid output format.")
            continue

        try:
            total_wallclock_ms = float(output_lines[0])
            ttft_ms = float(output_lines[1])
            instructions = int(output_lines[2])
            cycles = int(output_lines[3])
            mem_accesses = int(output_lines[4]) # Placeholder (0)
            output_tokens = [int(t) for t in output_lines[5].split(",") if t.strip()]
        except Exception as e:
            print(f"  [Error] Parsing failed: {e}")
            continue

        # Calculate Time Per Token (Decoding Phase)
        # (Total Time - TTFT) / (Total Tokens - 1) gives average time for remaining tokens
        decoding_time_ms = total_wallclock_ms - ttft_ms
        tokens_decoded = output_token_count - 1
        time_per_token = (decoding_time_ms / tokens_decoded) if tokens_decoded > 0 else 0.0

        generated_text = tokenizer.decode(output_tokens)

        with open(OUTPUT_FILE, "a") as f:
            f.write(f"--- {config['name']} ---\n")
            f.write(f"Settings: MatMul={'RVV' if config['matmul'] else 'Scalar'}, "
                    f"LN={'RVV' if config['ln'] else 'Scalar'}, "
                    f"Add={'RVV' if config['add'] else 'Scalar'}\n")
            f.write(f"Total Cycles:         {cycles:,}\n")
            f.write(f"Instructions:         {instructions:,}\n")
            f.write(f"TTFT (Latency):       {ttft_ms:.4f} ms\n")
            f.write(f"Time/Token (Decode):  {time_per_token:.4f} ms/token\n")
            f.write(f"Total Wallclock:      {total_wallclock_ms:.4f} ms\n")
            f.write(f"Output:               \"{generated_text}\"\n\n")

        print(f"  [Result] TTFT: {ttft_ms:.2f}ms | Decode: {time_per_token:.2f}ms/t")

    print(f"\nDone. Results in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()