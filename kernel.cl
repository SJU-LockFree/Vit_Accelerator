// kernel.cl

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define PATCH_SIZE 16
#define IMG_SIZE 224
#define IN_CHANS 3
#define EMBED_DIM 768
#define NUM_HEADS 12
#define HEAD_DIM 64 // 768 / 12
#define EPS 1e-6f
#define TS 16

// 1. Patch Embedding (Conv2d)
// Global Size: (total_patches * embed_dim) -> (196 * 768)
__kernel void conv2d_kernel(__global const half* input,   // ← float → half
    __global float* output,
    __global const float* weight,
    __global const float* bias)
{
    int idx = get_global_id(0);
    int output_size = IMG_SIZE / PATCH_SIZE; // 14
    int total_patches = output_size * output_size; // 196

    if (idx >= total_patches * EMBED_DIM) return;

    int oc = idx / total_patches;       // Output Channel (Embed Dim)
    int patch_idx = idx % total_patches;
    int oh = patch_idx / output_size;   // Patch Row
    int ow = patch_idx % output_size;   // Patch Col

    float sum = bias[oc];

    // 커널 윈도우 순회 (16x16x3)
    for (int ic = 0; ic < IN_CHANS; ++ic) {
        for (int kh = 0; kh < PATCH_SIZE; ++kh) {
            for (int kw = 0; kw < PATCH_SIZE; ++kw) {
                int ih = oh * PATCH_SIZE + kh;
                int iw = ow * PATCH_SIZE + kw;

                int input_idx = (ic * IMG_SIZE + ih) * IMG_SIZE + iw;
                int kernel_idx = ((oc * IN_CHANS + ic) * PATCH_SIZE + kh) * PATCH_SIZE + kw;

                // ★ half → float 변환해서 쓰기 ★
                float in_val = (float)(input[input_idx]);
                float w_val = weight[kernel_idx];

                sum += in_val * w_val;
            }
        }
    }

    // Output Index: patch_idx * EMBED_DIM + oc
    output[patch_idx * EMBED_DIM + oc] = sum;
}


// 2. Class Token 붙이기 + Position Embedding 더하기
// Global Size: (total_tokens * embed_dim) -> 197 * 768
__kernel void prepare_input_kernel(__global const float* patch_tokens,
    __global float* final_tokens,
    __global const float* cls_token,
    __global const float* pos_emb)
{
    int idx = get_global_id(0);
    int num_patches = (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE);
    int total_tokens = num_patches + 1;

    if (idx >= total_tokens * EMBED_DIM) return;

    int token_idx = idx / EMBED_DIM;
    int dim_idx = idx % EMBED_DIM;

    float val = 0.0f;

    if (token_idx == 0) {
        // Class Token
        val = cls_token[dim_idx];
    }
    else {
        // Patch Tokens (앞에 1칸 밀림)
        val = patch_tokens[(token_idx - 1) * EMBED_DIM + dim_idx];
    }

    // Add Position Embedding
    final_tokens[idx] = val + pos_emb[idx];
}

// 3. Layer Normalization
// Global Size: (tokens) -> 197
// 각 스레드가 1개의 토큰(768차원)을 담당하여 정규화 수행
__kernel void layer_norm_kernel(__global const float* input,
    __global float* output,
    __global const float* weight,
    __global const float* bias)
{
    int t = get_global_id(0); // token index
    int total_tokens = (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE) + 1;
    if (t >= total_tokens) return;

    int offset = t * EMBED_DIM;

    // 1) mean, var 계산
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = 0; i < EMBED_DIM; i++) {
        float v = input[offset + i];
        sum += v;
        sum_sq += v * v;
    }

    float mean = sum / (float)EMBED_DIM;
    float var = sum_sq / (float)EMBED_DIM - mean * mean;
    float invstd = 1.0f / sqrt(var + EPS);

    // 2) 정규화 + scale(gamma) + shift(beta)
    for (int i = 0; i < EMBED_DIM; i++) {
        float x = input[offset + i];
        float gamma = weight[i]; // 차원별
        float beta = bias[i];

        output[offset + i] = (x - mean) * invstd * gamma + beta;
    }
}


// 4. Linear Layer (Matrix Multiplication)
 //Global Size: (tokens, out_features)
// 4. Linear Layer (Matrix Multiplication, Tiled)
// Global Size: (tokens, out_features)
// Local Size : (TS, TS)  // TS = 16
__kernel void linear_kernel(__global const float* input,
    __global float* output,
    __global const float* weight,
    __global const float* bias,
    int in_features,
    int out_features,
    int tokens)
{
    // 전역 인덱스: 행(row)=token, 열(col)=out_feature
    int row = get_global_id(0);   // token index
    int col = get_global_id(1);   // out feature index

    // 워크그룹 내 로컬 인덱스
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);

    // 타일 크기
    const int TSz = TS; // 16

    // 로컬 메모리 타일
    __local float As[TS][TS];  // input 타일  : [token_tile, k_tile]
    __local float Bs[TS][TS];  // weight 타일 : [k_tile, out_tile]

    float sum = 0.0f;

    // K 방향(in_features)으로 몇 개의 타일이 필요한지
    int num_tiles = (in_features + TSz - 1) / TSz;

    for (int t = 0; t < num_tiles; ++t) {
        int k_base = t * TSz;

        // ---- A 타일 로딩: input[row, k_base + local_col] ----
        int a_col = k_base + local_col;
        if (row < tokens && a_col < in_features)
            As[local_row][local_col] = input[row * in_features + a_col];
        else
            As[local_row][local_col] = 0.0f;

        // ---- B 타일 로딩: weight[col, k_base + local_row] ----
        // weight 레이아웃: [out_features, in_features]
        int b_row = k_base + local_row; // in_feature index
        if (col < out_features && b_row < in_features)
            Bs[local_row][local_col] = weight[col * in_features + b_row];
        else
            Bs[local_row][local_col] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        // ---- 타일 내 곱셈/누적 ----
        // As: [local_row][k], Bs: [k][local_col]
        for (int k = 0; k < TSz; ++k) {
            sum += As[local_row][k] * Bs[k][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 범위 안일 때만 결과 저장
    if (row < tokens && col < out_features) {
        output[row * out_features + col] = sum + bias[col];
    }
}


// 5. Add Residual (Element-wise add)
// Global Size: (total_elements)
__kernel void add_kernel(__global const float* input,
    __global float* output)
{
    int idx = get_global_id(0);
    output[idx] = output[idx] + input[idx];
}

// 6. GELU Activation
// Global Size: (total_elements)
__kernel void gelu_kernel(__global float* data)
{
    int idx = get_global_id(0);
    float x = data[idx];
    data[idx] = 0.5f * x * (1.0f + erf(x * 0.70710678f));
}


// 7. Attention Score Calculation (Q * K^T)
// Global Size: (heads, tokens, tokens)
// Heads: 12, Tokens: 197
__kernel void attn_score_kernel(__global const float* qkv,
    __global float* scores,
    int total_tokens)
{
    int h = get_global_id(0); // head
    int i = get_global_id(1); // query token
    int j = get_global_id(2); // key token

    if (h >= NUM_HEADS || i >= total_tokens || j >= total_tokens) return;

    // 한 토큰당 3 * EMBED_DIM float
    int stride = 3 * EMBED_DIM;

    float score = 0.0f;
    for (int d = 0; d < HEAD_DIM; d++) {
        // Q: offset 0 ~ EMBED_DIM-1
        int q_idx = i * stride + (h * HEAD_DIM + d);

        // K: offset EMBED_DIM ~ 2*EMBED_DIM-1
        int k_idx = j * stride + (EMBED_DIM + h * HEAD_DIM + d);

        score += qkv[q_idx] * qkv[k_idx];
    }

    scores[(h * total_tokens + i) * total_tokens + j] =
        score / sqrt((float)HEAD_DIM);
}


// 8. Softmax (Applied per row in scores)
// Global Size: (heads, tokens)
// 각 스레드가 하나의 행(row)을 맡아서 Softmax 처리
__kernel void softmax_kernel(__global float* scores, int total_tokens)
{
    int h = get_global_id(0);
    int i = get_global_id(1);

    if (h >= NUM_HEADS || i >= total_tokens) return;

    int row_offset = (h * total_tokens + i) * total_tokens;

    float max_val = scores[row_offset];
    for (int j = 1; j < total_tokens; j++) {
        float v = scores[row_offset + j];
        if (v > max_val) max_val = v;
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < total_tokens; j++) {
        float ev = exp(scores[row_offset + j] - max_val);
        scores[row_offset + j] = ev;
        sum_exp += ev;
    }

    float inv_sum = 1.0f / sum_exp;
    for (int j = 0; j < total_tokens; j++) {
        scores[row_offset + j] *= inv_sum;
    }
}

// 9. Attention Value Calculation (Scores * V)
// Global Size: (heads, tokens, head_dim)
__kernel void attn_value_kernel(__global const float* scores,
    __global const float* qkv,
    __global float* attn_output,
    int total_tokens)
{
    int h = get_global_id(0); // head
    int i = get_global_id(1); // query token
    int d = get_global_id(2); // head dim

    if (h >= NUM_HEADS || i >= total_tokens || d >= HEAD_DIM) return;

    int row_offset = (h * total_tokens + i) * total_tokens;
    int stride = 3 * EMBED_DIM;

    float sum = 0.0f;
    for (int j = 0; j < total_tokens; j++) {
        float s = scores[row_offset + j];

        // V: offset 2*EMBED_DIM ~ 3*EMBED_DIM-1
        int v_idx = j * stride + (2 * EMBED_DIM + h * HEAD_DIM + d);

        sum += s * qkv[v_idx];
    }

    // 최종 [tokens, EMBED_DIM]로 저장
    attn_output[i * EMBED_DIM + h * HEAD_DIM + d] = sum;
}
