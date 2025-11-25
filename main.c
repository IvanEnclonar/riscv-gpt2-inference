#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>


//    ABLATION CONFIGURATION
//    1 = Use RISC-V Vector (RVV) Implementation
//    0 = Use Scalar Baseline Implementation

#define ENABLE_RVV_MATMUL     1 
#define ENABLE_RVV_LAYERNORM  0 
#define ENABLE_RVV_ADD        0

#if (ENABLE_RVV_MATMUL || ENABLE_RVV_LAYERNORM || ENABLE_RVV_ADD )
    #include <riscv_vector.h>
    #define HAS_RVV_HEADER 1
#endif

#define N_LAYERS 12
#define D_MODEL 768
#define N_HEADS 12
#define HEAD_SIZE 64   // 768 / 12
#define VOCAB_SIZE 50257
#define MAX_SEQ_LEN 1024
#define EPS 1e-5f

#define SQRT_2_PI 0.7978845608f
#define C_GELU 0.044715f

typedef struct {
    float *wte; // [VOCAB, D_MODEL]
    float *wpe; // [MAX_SEQ, D_MODEL]
    float *ln1_w[N_LAYERS], *ln1_b[N_LAYERS];
    float *attn_w[N_LAYERS], *attn_b[N_LAYERS];
    float *attn_proj_w[N_LAYERS], *attn_proj_b[N_LAYERS];
    float *ln2_w[N_LAYERS], *ln2_b[N_LAYERS];
    float *mlp_fc_w[N_LAYERS], *mlp_fc_b[N_LAYERS];
    float *mlp_proj_w[N_LAYERS], *mlp_proj_b[N_LAYERS];
    float *ln_f_w, *ln_f_b;
    float *lm_head;
} GPT2Weights;

typedef struct {
    float *key_cache;   // [Layers * Heads * Seq * HeadSize]
    float *value_cache; // [Layers * Heads * Seq * HeadSize]
} GPT2State;


// SCALAR KERNELS (BASELINE)
void add_scalar(float *out, float *a, float *b, int size) {
    for(int i=0; i<size; i++) out[i] = a[i] + b[i];
}

void gelu_scalar(float *x, int size) {
    for(int i=0; i<size; i++) {
        float xv = x[i];
        float cube = C_GELU * xv * xv * xv;
        float inner = SQRT_2_PI * (xv + cube);
        x[i] = 0.5f * xv * (1.0f + tanhf(inner));
    }
}

void layernorm_scalar(float *out, float *x, float *g, float *b, int size) {
    float mean = 0.0f;
    for(int i=0; i<size; i++) mean += x[i];
    mean /= size;
    
    float var = 0.0f;
    for(int i=0; i<size; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= size;
    
    float inv_std = 1.0f / sqrtf(var + EPS);
    for(int i=0; i<size; i++) {
        out[i] = (x[i] - mean) * inv_std * g[i] + b[i];
    }
}

void matmul_scalar(float *out, float *x, float *w, float *b, int dim_in, int dim_out) {
    for (int i = 0; i < dim_out; i++) {
        float val = (b != NULL) ? b[i] : 0.0f;
        for (int j = 0; j < dim_in; j++) {
            val += x[j] * w[j * dim_out + i];
        }
        out[i] = val;
    }
}


// RISC-V VECTOR KERNELS
#ifdef HAS_RVV_HEADER

void add_rvv(float *out, float *a, float *b, int size) {
    size_t vl;
    for (int i = 0; i < size; i += vl) {
        vl = __riscv_vsetvl_e32m8(size - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vl);
        vfloat32m8_t vres = __riscv_vfadd_vv_f32m8(va, vb, vl);
        __riscv_vse32_v_f32m8(out + i, vres, vl);
    }
}

void layernorm_rvv(float *out, float *x, float *g, float *b, int size) {
    size_t vl;
    // Mean
    vfloat32m1_t v_sum = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    int ptr = 0;
    while(ptr < size) {
        vl = __riscv_vsetvl_e32m8(size - ptr);
        vfloat32m8_t v_data = __riscv_vle32_v_f32m8(x + ptr, vl);
        v_sum = __riscv_vfredusum_vs_f32m8_f32m1(v_data, v_sum, vl);
        ptr += vl;
    }
    float mean = __riscv_vfmv_f_s_f32m1_f32(v_sum) / size;

    // Variance
    vfloat32m1_t v_var = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    ptr = 0;
    while(ptr < size) {
        vl = __riscv_vsetvl_e32m8(size - ptr);
        vfloat32m8_t v_data = __riscv_vle32_v_f32m8(x + ptr, vl);
        vfloat32m8_t v_diff = __riscv_vfsub_vf_f32m8(v_data, mean, vl);
        vfloat32m8_t v_sq = __riscv_vfmul_vv_f32m8(v_diff, v_diff, vl);
        v_var = __riscv_vfredusum_vs_f32m8_f32m1(v_sq, v_var, vl);
        ptr += vl;
    }
    float var = __riscv_vfmv_f_s_f32m1_f32(v_var) / size;
    float inv_std = 1.0f / sqrtf(var + EPS);

    // Normalize
    ptr = 0;
    while(ptr < size) {
        vl = __riscv_vsetvl_e32m8(size - ptr);
        vfloat32m8_t v_data = __riscv_vle32_v_f32m8(x + ptr, vl);
        vfloat32m8_t v_g = __riscv_vle32_v_f32m8(g + ptr, vl);
        vfloat32m8_t v_b = __riscv_vle32_v_f32m8(b + ptr, vl);
        
        vfloat32m8_t v_norm = __riscv_vfsub_vf_f32m8(v_data, mean, vl);
        v_norm = __riscv_vfmul_vf_f32m8(v_norm, inv_std, vl);
        v_norm = __riscv_vfmacc_vv_f32m8(v_b, v_norm, v_g, vl);
        
        __riscv_vse32_v_f32m8(out + ptr, v_norm, vl);
        ptr += vl;
    }
}

void matmul_rvv(float *out, float *x, float *w, float *b, int dim_in, int dim_out) {
    size_t vl;
    for (int i = 0; i < dim_out; i += vl) {
        vl = __riscv_vsetvl_e32m8(dim_out - i);
        vfloat32m8_t v_acc;
        if (b != NULL) v_acc = __riscv_vle32_v_f32m8(b + i, vl);
        else v_acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);

        for (int j = 0; j < dim_in; j++) {
            float scalar_x = x[j];
            vfloat32m8_t v_w = __riscv_vle32_v_f32m8(w + (j * dim_out + i), vl);
            v_acc = __riscv_vfmacc_vf_f32m8(v_acc, scalar_x, v_w, vl);
        }
        __riscv_vse32_v_f32m8(out + i, v_acc, vl);
    }
}
#endif


// Mapping
#if ENABLE_RVV_MATMUL
    #define MATMUL matmul_rvv
    const char* MODE_MATMUL = "RVV";
#else
    #define MATMUL matmul_scalar
    const char* MODE_MATMUL = "Scalar";
#endif

#if ENABLE_RVV_LAYERNORM
    #define LAYERNORM layernorm_rvv
    const char* MODE_LN = "RVV";
#else
    #define LAYERNORM layernorm_scalar
    const char* MODE_LN = "Scalar";
#endif

#define GELU gelu_scalar

#if ENABLE_RVV_ADD
    #define ADD add_rvv
    const char* MODE_ADD = "RVV";
#else
    #define ADD add_scalar
    const char* MODE_ADD = "Scalar";
#endif


void softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

void attention(float *out, float *x, 
               float *c_attn_w, float *c_attn_b, 
               float *c_proj_w, float *c_proj_b,
               GPT2State *state, int layer, int pos) {
    
    // 1. QKV Projection (Mapped)
    float qkv[3 * D_MODEL]; 
    MATMUL(qkv, x, c_attn_w, c_attn_b, D_MODEL, 3 * D_MODEL);

    float *q = qkv;
    float *k = qkv + D_MODEL;
    float *v = qkv + 2 * D_MODEL;

    float att_out[D_MODEL];

    // 2. Multi-Head Attention Loop
    for (int h = 0; h < N_HEADS; h++) {
        float *head_q = q + h * HEAD_SIZE;
        
        int cache_offset = layer * (N_HEADS * MAX_SEQ_LEN * HEAD_SIZE) + 
                           h * (MAX_SEQ_LEN * HEAD_SIZE) + 
                           pos * HEAD_SIZE;
        
        float *cache_k = state->key_cache + cache_offset;
        float *cache_v = state->value_cache + cache_offset;

        memcpy(cache_k, k + h * HEAD_SIZE, HEAD_SIZE * sizeof(float));
        memcpy(cache_v, v + h * HEAD_SIZE, HEAD_SIZE * sizeof(float));

        float scores[MAX_SEQ_LEN]; 
        
        // --- SCORE CALCULATION ---
        // For strict ablation, we could vectorize this Dot Product too.
        // For this demo, we keep it manual/scalar loop to focus on the big kernels.
        for (int t = 0; t <= pos; t++) {
            int past_offset = layer * (N_HEADS * MAX_SEQ_LEN * HEAD_SIZE) + 
                              h * (MAX_SEQ_LEN * HEAD_SIZE) + 
                              t * HEAD_SIZE;
            float *past_k = state->key_cache + past_offset;

            float score = 0.0f;
            for (int i = 0; i < HEAD_SIZE; i++) score += head_q[i] * past_k[i];
            score /= sqrtf((float)HEAD_SIZE);
            scores[t] = score;
        }

        softmax(scores, pos + 1);

        // --- WEIGHTED SUM ---
        float *head_out = att_out + h * HEAD_SIZE;
        for (int i = 0; i < HEAD_SIZE; i++) head_out[i] = 0.0f;

        for (int t = 0; t <= pos; t++) {
            int past_offset = layer * (N_HEADS * MAX_SEQ_LEN * HEAD_SIZE) + 
                              h * (MAX_SEQ_LEN * HEAD_SIZE) + 
                              t * HEAD_SIZE;
            float *past_v = state->value_cache + past_offset;
            float prob = scores[t];
            for (int i = 0; i < HEAD_SIZE; i++) head_out[i] += prob * past_v[i];
        }
    }

    // 3. Output Projection (Mapped)
    MATMUL(out, att_out, c_proj_w, c_proj_b, D_MODEL, D_MODEL);
}

void transformer_block(float *x, GPT2Weights *w, GPT2State *s, int layer, int pos) {
    float resid[D_MODEL];
    memcpy(resid, x, D_MODEL * sizeof(float));

    // LN 1
    float ln1[D_MODEL];
    LAYERNORM(ln1, x, w->ln1_w[layer], w->ln1_b[layer], D_MODEL);

    // Attention
    float attn_out[D_MODEL];
    attention(attn_out, ln1, w->attn_w[layer], w->attn_b[layer], 
              w->attn_proj_w[layer], w->attn_proj_b[layer], s, layer, pos);
    
    // Resid 1
    ADD(x, resid, attn_out, D_MODEL);
    memcpy(resid, x, D_MODEL * sizeof(float));

    // LN 2
    float ln2[D_MODEL];
    LAYERNORM(ln2, x, w->ln2_w[layer], w->ln2_b[layer], D_MODEL);

    // MLP FC
    float mlp_hidden[4 * D_MODEL];
    MATMUL(mlp_hidden, ln2, w->mlp_fc_w[layer], w->mlp_fc_b[layer], D_MODEL, 4 * D_MODEL);
    
    // GELU
    GELU(mlp_hidden, 4 * D_MODEL);
    
    // MLP Proj
    float mlp_out[D_MODEL];
    MATMUL(mlp_out, mlp_hidden, w->mlp_proj_w[layer], w->mlp_proj_b[layer], 4 * D_MODEL, D_MODEL);

    // Resid 2
    ADD(x, resid, mlp_out, D_MODEL);
}

int main() {
    printf("ABLATIONS:\n");
    printf("MatMul:    %s\n", MODE_MATMUL);
    printf("LayerNorm: %s\n", MODE_LN);
    printf("Add:       %s\n", MODE_ADD);
    
    FILE *f = fopen("gpt2_weights.bin", "rb");
    if (!f) { printf("Error: gpt2_weights.bin not found\n"); return 1; }
    fseek(f, 0, SEEK_END);
    long filesize = ftell(f);
    fseek(f, 0, SEEK_SET);
    float *memory = (float*)malloc(filesize);
    if(!memory) { printf("Malloc failed\n"); return 1; }
    
    if (fread(memory, 1, filesize, f) != filesize) {
        printf("Error reading weights\n");
        free(memory); fclose(f); return 1;
    }
    fclose(f);

    GPT2Weights w;
    float *ptr = memory;
    w.wte = ptr; ptr += VOCAB_SIZE * D_MODEL;
    w.wpe = ptr; ptr += MAX_SEQ_LEN * D_MODEL;
    for (int i = 0; i < N_LAYERS; i++) {
        w.ln1_w[i] = ptr; ptr += D_MODEL;
        w.ln1_b[i] = ptr; ptr += D_MODEL;
        w.attn_w[i] = ptr; ptr += D_MODEL * 3 * D_MODEL;
        w.attn_b[i] = ptr; ptr += 3 * D_MODEL;
        w.attn_proj_w[i] = ptr; ptr += D_MODEL * D_MODEL;
        w.attn_proj_b[i] = ptr; ptr += D_MODEL;
        w.ln2_w[i] = ptr; ptr += D_MODEL;
        w.ln2_b[i] = ptr; ptr += D_MODEL;
        w.mlp_fc_w[i] = ptr; ptr += D_MODEL * 4 * D_MODEL;
        w.mlp_fc_b[i] = ptr; ptr += 4 * D_MODEL;
        w.mlp_proj_w[i] = ptr; ptr += 4 * D_MODEL * D_MODEL;
        w.mlp_proj_b[i] = ptr; ptr += D_MODEL;
    }
    w.ln_f_w = ptr; ptr += D_MODEL;
    w.ln_f_b = ptr; ptr += D_MODEL;
    w.lm_head = ptr;

    GPT2State state;
    long cache_size = (long)N_LAYERS * MAX_SEQ_LEN * D_MODEL; 
    state.key_cache = (float*)malloc(cache_size * sizeof(float));
    state.value_cache = (float*)malloc(cache_size * sizeof(float));

    // Prompt: "The quick brown fox jumps over the lazy"
    int prompt_tokens[] = { 464, 2068, 7586, 21831, 18045, 625, 262, 16931 };
    int num_prompt = sizeof(prompt_tokens) / sizeof(int);
    int tokens_to_generate = 10; 

    printf("Prompt Length: %d. Generating %d tokens.\n", num_prompt, tokens_to_generate);

    int current_token = prompt_tokens[0];
    int pos = 0;
    float x[D_MODEL];
    float *logits = (float*)malloc(VOCAB_SIZE * sizeof(float));
    
    clock_t start = clock();

    while (pos < num_prompt + tokens_to_generate) {
        // Embedding
        for(int i=0; i<D_MODEL; i++) {
            x[i] = w.wte[current_token * D_MODEL + i] + w.wpe[pos * D_MODEL + i];
        }

        // Forward
        for(int i=0; i<N_LAYERS; i++) {
            transformer_block(x, &w, &state, i, pos);
        }

        // Final Norm
        float final_norm[D_MODEL];
        LAYERNORM(final_norm, x, w.ln_f_w, w.ln_f_b, D_MODEL);

        // Next Token Logic
        int next_token;
        if (pos < num_prompt - 1) {
            next_token = prompt_tokens[pos + 1];
        } else {
            // Logits
            MATMUL(logits, final_norm, w.lm_head, NULL, D_MODEL, VOCAB_SIZE);
            
            float max_prob = -1e9;
            int argmax = 0;
            for(int i=0; i<VOCAB_SIZE; i++) {
                if (logits[i] > max_prob) {
                    max_prob = logits[i];
                    argmax = i;
                }
            }
            next_token = argmax;
            printf("Step %d | Token: %d\n", pos, next_token);
        }

        pos++;
        current_token = next_token;
        if (pos >= MAX_SEQ_LEN) break;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Inference finished in %f seconds.\n", time_spent);

    free(memory);
    free(state.key_cache);
    free(state.value_cache);
    free(logits);
    return 0;
}