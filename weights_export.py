import numpy as np
import os
from transformers import GPT2LMHeadModel

model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
sd = model_hf.state_dict()

# 1. Define the specific order of tensors we want to write
keys = [
    "transformer.wte.weight",
    "transformer.wpe.weight",
]

# Add blocks
for i in range(12):
    prefix = f"transformer.h.{i}."
    keys.extend([
        prefix + "ln_1.weight", prefix + "ln_1.bias",
        prefix + "attn.c_attn.weight", prefix + "attn.c_attn.bias",
        prefix + "attn.c_proj.weight", prefix + "attn.c_proj.bias",
        prefix + "ln_2.weight", prefix + "ln_2.bias",
        prefix + "mlp.c_fc.weight", prefix + "mlp.c_fc.bias",
        prefix + "mlp.c_proj.weight", prefix + "mlp.c_proj.bias"
    ])

# Final layers
keys.extend([
    "transformer.ln_f.weight",
    "transformer.ln_f.bias",
    "lm_head.weight" # Note: We will transpose this in C, or simpler: transpose here
])

# 2. Flatten and write to file
with open("gpt2_weights.bin", "wb") as f:
    for k in keys:
        t = sd[k].cpu().numpy()
        # CRITICAL: Transpose weights that are Conv1D in HF (Linear in standard terminology)
        # HF stores weights as (768, 3*768) for attn, but usually we want (In, Out) logic
        # For this C code, we will stick to row-major logic.
        # HF weights for Linear layers are usually (In_Features, Out_Features).
        # BUT lm_head is (Vocab, Emb). We want (Emb, Vocab) for the final matmul.
        if k == "lm_head.weight":
            t = t.T

        print(f"Writing {k} | Shape: {t.shape}")
        f.write(t.astype(np.float32).tobytes())

print("Saved to gpt2_weights.bin")