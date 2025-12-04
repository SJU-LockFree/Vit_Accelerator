// kernel.cl (Conv2d Layout Fix Version)

#define PATCH_SIZE 16
#define IMG_SIZE 224
#define IN_CHANS 3
#define EMBED_DIM 768
#define NUM_HEADS 12
#define HEAD_DIM 64 // 768 / 12
#define EPS 1e-6f
#define TS 16

// 1. Patch Embedding (Conv2d) [Layout 수정됨: Dim,Patch -> Patch,Dim]
__kernel void conv2d_kernel(__global const float* input,
    __global float* output,
    __global const float* weight,
    __global const float* bias)
{
    int idx = get_global_id(0);

    int output_size = IMG_SIZE / PATCH_SIZE; // 14
    int patches_per_img = output_size * output_size; // 196
    int dims_per_img = patches_per_img * EMBED_DIM; // 196 * 768

    // 현재 스레드가 담당하는 인덱스 해석
    // (기존 GWS 구조상 oc가 상위, patch가 하위로 분해됨)
    int batch_idx = idx / dims_per_img;
    int idx_in_batch = idx % dims_per_img;

    int oc = idx_in_batch / patches_per_img;      // Feature Dim Index (0~767)
    int patch_idx = idx_in_batch % patches_per_img; // Patch Index (0~195)

    int oh = patch_idx / output_size;
    int ow = patch_idx % output_size;

    float sum = bias[oc];

    // 입력 오프셋: 배치마다 이미지 크기만큼 점프
    int input_offset = batch_idx * (3 * IMG_SIZE * IMG_SIZE);

#pragma unroll
    for (int ic = 0; ic < IN_CHANS; ++ic) {
        for (int kh = 0; kh < PATCH_SIZE; ++kh) {
            for (int kw = 0; kw < PATCH_SIZE; ++kw) {
                int ih = oh * PATCH_SIZE + kh;
                int iw = ow * PATCH_SIZE + kw;
                int input_idx = input_offset + (ic * IMG_SIZE + ih) * IMG_SIZE + iw;

                // Weight: [OC, IC, KH, KW]
                int kernel_idx = ((oc * IN_CHANS + ic) * PATCH_SIZE + kh) * PATCH_SIZE + kw;
                sum += input[input_idx] * weight[kernel_idx];
            }
        }
    }

    // [중요 수정] Transformer 입력 순서인 [Batch, Patch, Dim]으로 저장
    // 기존: output[idx] = sum; (이건 [Dim, Patch] 순서였음)

    int dest_idx = batch_idx * dims_per_img + patch_idx * EMBED_DIM + oc;
    output[dest_idx] = sum;
}

// 2. Class Token + Position Embedding
__kernel void prepare_input_kernel(__global const float* patch_tokens,
    __global float* final_tokens,
    __global const float* cls_token,
    __global const float* pos_emb)
{
    int idx = get_global_id(0);

    int num_patches = (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE); // 196
    int tokens_per_img = num_patches + 1; // 197
    int items_per_img = tokens_per_img * EMBED_DIM;

    int batch_idx = idx / items_per_img;
    int idx_in_img = idx % items_per_img; // 0 ~ (197*768 - 1)

    int token_idx = idx_in_img / EMBED_DIM;
    int dim_idx = idx_in_img % EMBED_DIM;

    float val = 0.0f;

    if (token_idx == 0) {
        val = cls_token[dim_idx];
    }
    else {
        // [확인] 여기서 patch_idx * EMBED_DIM 순서로 읽으므로,
        // 위 conv2d_kernel에서 저장 순서를 맞춰주어야 했음. (이제 맞음)
        int patch_offset = batch_idx * (num_patches * EMBED_DIM);
        val = patch_tokens[patch_offset + (token_idx - 1) * EMBED_DIM + dim_idx];
    }

    final_tokens[idx] = val + pos_emb[idx_in_img];
}

// 3. Layer Normalization
__kernel void layer_norm_kernel(__global const float* input,
    __global float* output,
    __constant const float* weight,
    __constant const float* bias)
{
    int t = get_global_id(0);
    int offset = t * EMBED_DIM;

    __global const float4* in_vec = (__global const float4*)(input + offset);
    __global float4* out_vec = (__global float4*)(output + offset);
    __constant const float4* gamma_vec = (__constant const float4*)weight;
    __constant const float4* beta_vec = (__constant const float4*)bias;

    float4 v_sum = (float4)(0.0f);
    float4 v_sum_sq = (float4)(0.0f);

#pragma unroll 
    for (int i = 0; i < EMBED_DIM / 4; i++) {
        float4 val = in_vec[i];
        v_sum += val;
        v_sum_sq += val * val;
    }

    float sum = v_sum.x + v_sum.y + v_sum.z + v_sum.w;
    float sum_sq = v_sum_sq.x + v_sum_sq.y + v_sum_sq.z + v_sum_sq.w;

    float mean = sum / (float)EMBED_DIM;
    float var = sum_sq / (float)EMBED_DIM - mean * mean;
    float invstd = 1.0f / sqrt(var + EPS);

#pragma unroll
    for (int i = 0; i < EMBED_DIM / 4; i++) {
        float4 val = in_vec[i];
        float4 gamma = gamma_vec[i];
        float4 beta = beta_vec[i];
        out_vec[i] = (val - mean) * invstd * gamma + beta;
    }
}

// 4. Linear Layer
__kernel void linear_kernel(__global const float* input,
    __global float* output,
    __global const float* weight,
    __constant const float* bias,
    int in_features,
    int out_features,
    int tokens)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);
    const int TSz = TS;

    __local float As[TS][TS];
    __local float Bs[TS][TS];

    float sum = 0.0f;
    int num_tiles = (in_features + TSz - 1) / TSz;

#pragma unroll
    for (int t = 0; t < num_tiles; ++t) {
        int k_base = t * TSz;
        int a_col = k_base + local_col;
        if (row < tokens && a_col < in_features)
            As[local_row][local_col] = input[row * in_features + a_col];
        else
            As[local_row][local_col] = 0.0f;

        int b_row = k_base + local_row;
        if (col < out_features && b_row < in_features)
            Bs[local_row][local_col] = weight[col * in_features + b_row];
        else
            Bs[local_row][local_col] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TSz; ++k) {
            sum += As[local_row][k] * Bs[k][local_col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < tokens && col < out_features) {
        output[row * out_features + col] = sum + bias[col];
    }
}

// 5. Add Residual
__kernel void add_kernel(__global const float* input, __global float* output)
{
    int idx = get_global_id(0);
    // output(buf1) = output(buf1) + input(residual)
    output[idx] = output[idx] + input[idx];
}

// 6. GELU
__kernel void gelu_kernel(__global float* data)
{
    int idx = get_global_id(0);
    float x = data[idx];
    data[idx] = 0.5f * x * (1.0f + erf(x * 0.70710678f));
}

// 7. Attention Score
__kernel void attn_score_kernel(__global const float* qkv,
    __global float* scores,
    int tokens_per_img)
{
    int global_h = get_global_id(0);
    int i = get_global_id(1);
    int j = get_global_id(2);

    int batch_idx = global_h / NUM_HEADS;
    int h = global_h % NUM_HEADS;

    if (i >= tokens_per_img || j >= tokens_per_img) return;

    int stride = 3 * EMBED_DIM;
    int qkv_batch_offset = batch_idx * (tokens_per_img * stride);
    int score_batch_offset = batch_idx * (NUM_HEADS * tokens_per_img * tokens_per_img);

    float4 score_vec = (float4)(0.0f);

#pragma unroll 4
    for (int d = 0; d < HEAD_DIM; d += 4) {
        int q_idx = qkv_batch_offset + i * stride + (h * HEAD_DIM + d);
        int k_idx = qkv_batch_offset + j * stride + (EMBED_DIM + h * HEAD_DIM + d);

        float4 vec_q = vload4(0, &qkv[q_idx]);
        float4 vec_k = vload4(0, &qkv[k_idx]);
        score_vec += vec_q * vec_k;
    }

    float final_score = score_vec.x + score_vec.y + score_vec.z + score_vec.w;
    int out_idx = score_batch_offset + (h * tokens_per_img + i) * tokens_per_img + j;
    scores[out_idx] = final_score / sqrt((float)HEAD_DIM);
}

// 8. Softmax
__kernel void softmax_kernel(__global float* scores, int tokens_per_img)
{
    int global_h = get_global_id(0);
    int i = get_global_id(1);

    if (i >= tokens_per_img) return;

    int row_offset = global_h * tokens_per_img * tokens_per_img + i * tokens_per_img;

    float max_val = scores[row_offset];
    for (int j = 1; j < tokens_per_img; j++) {
        float val = scores[row_offset + j];
        if (val > max_val) max_val = val;
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < tokens_per_img; j++) {
        float val = exp(scores[row_offset + j] - max_val);
        scores[row_offset + j] = val;
        sum_exp += val;
    }

    float inv_sum = 1.0f / sum_exp;
    for (int j = 0; j < tokens_per_img; j++) {
        scores[row_offset + j] *= inv_sum;
    }
}

// 9. Attn Value
__kernel void attn_value_kernel(__global const float* scores,
    __global const float* qkv,
    __global float* attn_output,
    int tokens_per_img)
{
    int global_h = get_global_id(0);
    int i = get_global_id(1);
    int d_vec = get_global_id(2);
    int d = d_vec * 4;

    int batch_idx = global_h / NUM_HEADS;
    int h = global_h % NUM_HEADS;

    if (i >= tokens_per_img || d >= HEAD_DIM) return;

    int stride = 3 * EMBED_DIM;
    int qkv_batch_offset = batch_idx * (tokens_per_img * stride);
    int score_row_offset = global_h * tokens_per_img * tokens_per_img + i * tokens_per_img;
    int out_batch_offset = batch_idx * (tokens_per_img * EMBED_DIM);
    int v_base_offset = 2 * EMBED_DIM + h * HEAD_DIM;

    float4 sum_vec = (float4)(0.0f);

#pragma unroll 4
    for (int j = 0; j < tokens_per_img; j++) {
        float s = scores[score_row_offset + j];
        int v_idx = qkv_batch_offset + j * stride + v_base_offset + d;
        float4 v_val = vload4(0, &qkv[v_idx]);
        sum_vec += s * v_val;
    }

    int out_idx = out_batch_offset + i * EMBED_DIM + h * HEAD_DIM + d;
    vstore4(sum_vec, 0, &attn_output[out_idx]);
}

// 10. Gather CLS Token
__kernel void gather_cls_kernel(__global const float* input_tokens,
    __global float* output_cls_vec,
    int tokens_per_img)
{
    int idx = get_global_id(0);
    int dim = EMBED_DIM;

    int batch_idx = idx / dim;
    int d = idx % dim;

    int input_idx = batch_idx * (tokens_per_img * dim) + d;

    output_cls_vec[idx] = input_tokens[input_idx];
}

#define LNQKV_WG_SIZE 64

__kernel void ln_qkv_fused_kernel(__global const float* input,
    __global float* qkv_out,
    __constant const float* gamma,
    __constant const float* beta,
    __global const float* weight,
    __constant const float* bias,
    int tokens)
{
    int token = get_group_id(0);     // 몇 번째 토큰인지 (row index)
    int lid = get_local_id(0);     // 워크그룹 내 로컬 ID
    int lsize = get_local_size(0);   // 워크그룹 크기

    if (token >= tokens) return;

    __local float x[EMBED_DIM];
    __local float norm[EMBED_DIM];
    __local float partial_sum[LNQKV_WG_SIZE];
    __local float partial_sq[LNQKV_WG_SIZE];
    __local float mean_var[2];

    // 1) 입력 토큰 로컬 메모리에 로드
    for (int d = lid; d < EMBED_DIM; d += lsize) {
        x[d] = input[token * EMBED_DIM + d];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2) mean/var 계산용 partial sum
    float s = 0.0f;
    float s2 = 0.0f;
    for (int d = lid; d < EMBED_DIM; d += lsize) {
        float v = x[d];
        s += v;
        s2 += v * v;
    }
    partial_sum[lid] = s;
    partial_sq[lid] = s2;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (int i = 0; i < lsize; i++) {
            sum += partial_sum[i];
            sum_sq += partial_sq[i];
        }
        float mean = sum / (float)EMBED_DIM;
        float var = sum_sq / (float)EMBED_DIM - mean * mean;
        float invstd = 1.0f / sqrt(var + EPS);
        mean_var[0] = mean;
        mean_var[1] = invstd;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float mean = mean_var[0];
    float invstd = mean_var[1];

    // 3) LayerNorm: norm[d] = (x-mean)/std * gamma + beta
    for (int d = lid; d < EMBED_DIM; d += lsize) {
        float v = x[d];
        float g = gamma[d];
        float b = beta[d];
        norm[d] = (v - mean) * invstd * g + b;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 4) QKV Linear: [tokens, EMBED_DIM] * [3*EMBED_DIM, EMBED_DIM]^T
    int out_features = 3 * EMBED_DIM;

    for (int col = lid; col < out_features; col += lsize) {
        float acc = bias[col];
        int w_base = col * EMBED_DIM;

        for (int d = 0; d < EMBED_DIM; ++d) {
            acc += norm[d] * weight[w_base + d];
        }

        qkv_out[token * out_features + col] = acc;
    }
}
