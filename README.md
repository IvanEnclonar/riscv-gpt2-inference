# RISCV GPT2 INFERENCE

This is a minimal, self-contained implementation of GPT-2 inference on RISC-V, designed to show how transformer-based language models can run on open hardware architectures without relying on large ML frameworks. The project focuses on the implementation of the RVV extension using C intrinsics, demonstrating how critical operations like Matrix Multiplication and Layer Normalization can be accelerated on embedded RISC-V CPUs through manual vectorization.

## Requirements
	•	RISC-V GNU toolchain (riscv64-linux-gnu-gcc)
	•	QEMU for RISC-V (qemu-riscv64) or a physical RISC-V board
	•	Run `python weights_export.py` first to download the GPT-2 weights and convert it into the expected flat binary format

## Build
```riscv64-linux-gnu-gcc -march=rv64gcv -mabi=lp64d -O3 -static -o gpt2_rvv main.c -lm```

## Run
```qemu-riscv64 ./gpt2_rvv```

---
<sub>Built in collaboration with [BorisVictoria](https://github.com/BorisVictoria) and [Joel Ethan Batac](https://github.com/marcusaurelys)</sub>
