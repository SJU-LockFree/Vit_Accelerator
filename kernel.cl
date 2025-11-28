// kernel.cl

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
__kernel void conv2d_kernel(__global const float* input,
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

                sum += input[input_idx] * weight[kernel_idx];
            }
        }
    }

    // ViT는 Flatten & Transpose를 하므로, (Patch, Channel) 순서로 저장해야 함
    // 원본 코드의 flatten_transpose 역할을 여기서 미리 수행
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
    int t = get_global_id(0); // Token Index
    // total_tokens는 인자로 받거나 상수로 정의
    int total_tokens = (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE) + 1;

    if (t >= total_tokens) return;

    float sum = 0.0f;
    float sum_sq = 0.0f;
    int offset = t * EMBED_DIM;

    // Mean & Variance Calculation
    for (int i = 0; i < EMBED_DIM; i++) {
        float val = input[offset + i];
        sum += val;
        sum_sq += val * val;
    }

    float mean = sum / EMBED_DIM;
    float var = sum_sq / EMBED_DIM - mean * mean;
    float inv_std = 1.0f / sqrt(var + EPS);

    // Normalize & Scale & Shift
    for (int i = 0; i < EMBED_DIM; i++) {
        output[offset + i] = (input[offset + i] - mean) * inv_std * weight[i] + bias[i];
    }
}

// 4. Linear Layer (Matrix Multiplication)
 //Global Size: (tokens, out_features)
__kernel void linear_kernel(__global const float* input,
    __global float* output,
    __global const float* weight,
    __global const float* bias,
    int in_features,
    int out_features,
    int tokens) // tokens 인자 필수!
{
    // Global ID: (row=Token, col=OutFeature)
    int row = get_global_id(0);
    int col = get_global_id(1);

    // Local ID
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);

    // Group ID (WorkGroup Index)
    int group_col = get_group_id(1); // Out Feature Block Index

    // Local Memory
    __local float As[TS][TS]; // Input Tile
    __local float Bs[TS][TS]; // Weight Tile

    float sum = 0.0f;
    int num_tiles = (in_features + TS - 1) / TS;

    for (int t = 0; t < num_tiles; t++) {
        // 1. Load Input Tile (As)
        // As[local_row][local_col] <--- Input[row][t*TS + local_col]
        int tiled_in_idx = t * TS + local_col;

        if (row < tokens && tiled_in_idx < in_features)
            As[local_row][local_col] = input[row * in_features + tiled_in_idx];
        else
            As[local_row][local_col] = 0.0f;

        // 2. Load Weight Tile (Bs)
        // Weight Shape: [Out_Features, In_Features]
        // 우리가 필요한 Weight Block: Rows(Out) = group_col*TS ~ +16, Cols(In) = t*TS ~ +16
        // 로딩 방식: 스레드들이 협력해서 Weight의 16x16 블록을 Bs에 복사
        // Bs[local_row][local_col] <--- Weight[Current_Block_Out_Row][Current_Block_In_Col]

        int w_out_row = group_col * TS + local_row; // Weight의 행 (Out Feature)
        int w_in_col = t * TS + local_col;         // Weight의 열 (In Feature)

        if (w_out_row < out_features && w_in_col < in_features)
            Bs[local_row][local_col] = weight[w_out_row * in_features + w_in_col];
        else
            Bs[local_row][local_col] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        // 3. Compute (Inner Product)
        // 내 Output 위치 (row, col)을 계산하기 위해
        // As의 내 row행 (local_row)과
        // Bs의 내 col행?? -> Bs는 [Out_Idx_Offset][In_Idx] 로 저장됨
        // 내 col(Out Feature)은 WorkGroup 내에서 'local_col' 번째임.
        // 하지만 위에서 로딩할 때 Bs[local_row][local_col]에 Weight[... + local_row][... + local_col] 넣음
        // 즉 Bs의 '행' 인덱스가 Out Feature 인덱스 오프셋임.
        // 내가 필요한 Weight 행은 Bs[local_col] 행임.

        for (int k = 0; k < TS; k++) {
            sum += As[local_row][k] * Bs[local_col][k];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

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
    // 0.5 * x * (1 + erf(x / sqrt(2)))
    data[idx] = 0.5f * x * (1.0f + erf(x * 0.70710678f));
}

// 7. Attention Score Calculation (Q * K^T)
// Global Size: (heads, tokens, tokens)
// Heads: 12, Tokens: 197
__kernel void attn_score_kernel(__global const float* qkv_input,
    __global float* scores,
    int total_tokens)
{
    int h = get_global_id(0); // Head index
    int i = get_global_id(1); // Query Token index
    int j = get_global_id(2); // Key Token index

    if (h >= NUM_HEADS || i >= total_tokens || j >= total_tokens) return;

    // Input layout assumed: [tokens, 3 * embed_dim] (from linear output)
    // Actually, user logic computes Q, K, V from separate parts of weight.
    // Let's assume input is QKV combined result: [tokens, 3, heads, head_dim] is hard to address.
    // Simplified: Input is (Tokens, 3 * Embed_Dim).
    // Q Start: 0, K Start: Embed_Dim, V Start: 2*Embed_Dim

    int q_offset = i * (3 * EMBED_DIM) + h * HEAD_DIM;
    int k_offset = j * (3 * EMBED_DIM) + EMBED_DIM + h * HEAD_DIM;

    float score = 0.0f;
    for (int d = 0; d < HEAD_DIM; d++) {
        score += qkv_input[q_offset + d] * qkv_input[k_offset + d];
    }

    scores[(h * total_tokens + i) * total_tokens + j] = score / sqrt((float)HEAD_DIM);
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

    // Max Find
    float max_val = scores[row_offset];
    for (int j = 1; j < total_tokens; j++) {
        float val = scores[row_offset + j];
        if (val > max_val) max_val = val;
    }

    // Exp & Sum
    float sum_exp = 0.0f;
    for (int j = 0; j < total_tokens; j++) {
        float val = exp(scores[row_offset + j] - max_val);
        scores[row_offset + j] = val;
        sum_exp += val;
    }

    // Normalize
    float inv_sum = 1.0f / sum_exp;
    for (int j = 0; j < total_tokens; j++) {
        scores[row_offset + j] *= inv_sum;
    }
}

// 9. Attention Value Calculation (Scores * V)
// Global Size: (heads, tokens, head_dim)
__kernel void attn_value_kernel(__global const float* scores,
    __global const float* qkv_input,
    __global float* attn_output,
    int total_tokens)
{
    int h = get_global_id(0);
    int i = get_global_id(1); // Output Token Index
    int d = get_global_id(2); // Head Dim Index

    if (h >= NUM_HEADS || i >= total_tokens || d >= HEAD_DIM) return;

    int v_base_offset = 2 * EMBED_DIM + h * HEAD_DIM; // V starts at 2*Embed
    int row_offset = (h * total_tokens + i) * total_tokens;

    float sum = 0.0f;
    for (int j = 0; j < total_tokens; j++) {
        float score = scores[row_offset + j];
        float v_val = qkv_input[j * (3 * EMBED_DIM) + v_base_offset + d];
        sum += score * v_val;
    }

    // Output should be flattened for next linear layer: [Tokens, Embed_Dim]
    // Embed_Dim = Heads * Head_Dim
    attn_output[i * EMBED_DIM + h * HEAD_DIM + d] = sum;
}

// 10. Final Classifier Softmax
// Global Size: (NUM_CLASSES) 가 아니라 (WorkGroupSize)로 잡아서 한 번에 처리
// Local Size: 256 or 512 (NUM_CLASSES가 1000이므로 루프를 약간 돕니다)
__kernel void final_softmax_kernel(__global float* logits,
    int num_classes)
{
    // 로컬 메모리: 워크 그룹 내 스레드들이 공유하는 고속 메모리
    // 1024 float = 4KB (충분함)
    __local float s_data[1024];

    int tid = get_local_id(0);
    int group_size = get_local_size(0);

    // 1. Load Data to Local Memory (Global -> Local)
    // 1000개 데이터를 스레드들이 나눠서 로딩
    float val = (tid < num_classes) ? logits[tid] : -INFINITY;
    s_data[tid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. Find Max (Parallel Reduction)
    // 수치 안정성을 위해 최대값을 찾아서 뺍니다.
    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_data[tid + stride] > s_data[tid]) {
                s_data[tid] = s_data[tid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_val = s_data[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3. Exponentiate
    // 다시 본래 값을 로드하고 exp 연산 수행
    val = (tid < num_classes) ? logits[tid] : -INFINITY;
    val = exp(val - max_val);
    s_data[tid] = (tid < num_classes) ? val : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    // 4. Sum (Parallel Reduction)
    // exp한 값들의 합을 구함
    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float sum = s_data[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // 5. Normalize & Write Back (Local -> Global)
    if (tid < num_classes) {
        logits[tid] = val / sum; // In-place update
    }
}

// [NEW] Fused Kernel: Linear + Bias + GELU
// Global Size: (tokens, out_features)
__kernel void linear_bias_gelu_kernel(__global const float* input,
    __global float* output,
    __global const float* weight,
    __global const float* bias,
    int in_features,
    int out_features)
{
    int t = get_global_id(0); // Token Index
    int o = get_global_id(1); // Output Feature Index

    if (o >= out_features) return;

    // 1. Bias Add (Initialize sum with bias)
    float sum = bias[o];
    int w_offset = o * in_features;
    int in_offset = t * in_features;

    // 2. Linear (Matrix Multiplication)
    for (int i = 0; i < in_features; i++) {
        sum += input[in_offset + i] * weight[w_offset + i];
    }

    // 3. GELU Activation (Fused!)
    // Formula: 0.5 * x * (1 + erf(x / sqrt(2)))
    // 1 / sqrt(2) approx 0.70710678f
    float gelu_val = 0.5f * sum * (1.0f + erf(sum * 0.70710678f));

    // 4. Store Result
    output[t * out_features + o] = gelu_val;
}