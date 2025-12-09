// kernel.cl

#define PATCH_SIZE 16
#define IMG_SIZE 224
#define IN_CHANS 3
#define EMBED_DIM 768
#define NUM_HEADS 12
#define HEAD_DIM 64 // 768 / 12
#define EPS 1e-6f
#define TS 16


// 1. Patch Embedding (Conv2d) - Vectorized Optimization
// Global Size: (total_patches * embed_dim) -> (196 * 768)
__kernel void conv2d_kernel(__global const float* input,
    __global float* output,
    __global const float* weight,
    __constant const float* bias,
    int batch_size)
{
    int idx = get_global_id(0);

    int out_hw = IMG_SIZE / PATCH_SIZE;              // 14
    int patches_per_img = out_hw * out_hw;                    // 196
    int dims_per_img = patches_per_img * EMBED_DIM;        // 196 * 768
    int total_elems = batch_size * dims_per_img;

    if (idx >= total_elems) return;

    int b = idx / dims_per_img;      // 배치 인덱스
    int idx_in_img = idx % dims_per_img;

    /*int oc = idx_in_img / patches_per_img;
    int patch_idx = idx_in_img % patches_per_img;*/
    int patch_idx = idx_in_img / EMBED_DIM;
    int oc = idx_in_img % EMBED_DIM;

    int oh = patch_idx / out_hw;
    int ow = patch_idx % out_hw;

    int start_y = oh * PATCH_SIZE;
    int start_x = ow * PATCH_SIZE;

    int input_batch_offset = b * (IN_CHANS * IMG_SIZE * IMG_SIZE);

    float sum = bias[oc];
    int weight_oc_offset = oc * IN_CHANS * PATCH_SIZE * PATCH_SIZE;

    for (int ic = 0; ic < IN_CHANS; ++ic) {
        int input_ic_offset = input_batch_offset + ic * IMG_SIZE * IMG_SIZE;
        int weight_ic_offset = weight_oc_offset + ic * PATCH_SIZE * PATCH_SIZE;

        for (int kh = 0; kh < PATCH_SIZE; ++kh) {
            int input_row_idx = input_ic_offset + (start_y + kh) * IMG_SIZE + start_x;
            int weight_row_idx = weight_ic_offset + kh * PATCH_SIZE;

            float4 in0 = vload4(0, &input[input_row_idx]);
            float4 w0 = vload4(0, &weight[weight_row_idx]);
            sum += dot(in0, w0);

            float4 in1 = vload4(1, &input[input_row_idx]);
            float4 w1 = vload4(1, &weight[weight_row_idx]);
            sum += dot(in1, w1);

            float4 in2 = vload4(2, &input[input_row_idx]);
            float4 w2 = vload4(2, &weight[weight_row_idx]);
            sum += dot(in2, w2);

            float4 in3 = vload4(3, &input[input_row_idx]);
            float4 w3 = vload4(3, &weight[weight_row_idx]);
            sum += dot(in3, w3);
        }
    }

    int dest_idx = b * dims_per_img + patch_idx * EMBED_DIM + oc;
    output[dest_idx] = sum;
}


// 2. Class Token 붙이기 + Position Embedding 더하기
// Global Size: (total_tokens * embed_dim) -> 197 * 768
__kernel void prepare_input_kernel(__global const float* patch_tokens,
    __global float* final_tokens,
    __global const float* cls_token,
    __global const float* pos_emb,
    int tokens_per_img,   // 197
    int batch_size)
{
    int idx = get_global_id(0);

    int total_tokens = tokens_per_img * batch_size;
    int total_elems = total_tokens * EMBED_DIM;

    if (idx >= total_elems) return;

    int token_idx_global = idx / EMBED_DIM;      // [0 .. total_tokens-1]
    int dim_idx = idx % EMBED_DIM;

    int b = token_idx_global / tokens_per_img;
    int token_idx = token_idx_global % tokens_per_img; // 0: CLS, 1~: patch

    float val = 0.0f;

    if (token_idx == 0) {
        val = cls_token[dim_idx];
    }
    else {
        int patch_idx = token_idx - 1;
        int src_idx = b * ((tokens_per_img - 1) * EMBED_DIM)
            + patch_idx * EMBED_DIM + dim_idx;
        val = patch_tokens[src_idx];
    }

    // pos_emb도 배치별 같은 걸 쓰고 싶다면, 보통 [tokens_per_img * EMBED_DIM] 크기로 두고
    int pos_idx = token_idx * EMBED_DIM + dim_idx;
    final_tokens[idx] = val + pos_emb[pos_idx];
}


// 3. Layer Normalization
// Global Size: (tokens) -> 197
// 각 스레드가 1개의 토큰(768차원)을 담당하여 정규화 수행
__kernel void layer_norm_kernel(__global const float* input,
    __global float* output,
    __constant const float* weight,     /* [ constant화 ] */
    __constant const float* bias)       /* [ constant화 ] */
{
    // 256 threads work together for one token
    int tid = get_local_id(0);       // 0 ~ 255
    int token_id = get_group_id(0);  // 0 ~ 196

    // EMBED_DIM = 768
    // 768 / 4 = 192 iterations needed for float4
    // Local Size 256 is enough to cover 192.

    // Shared Memory for Reduction
    __local float s_sum[256];
    __local float s_sq[256];
    __local float s_mean;
    __local float s_invstd;

    // 1. Load Data & Calculate Partial Sums
    float4 val = (float4)(0.0f);
    float my_sum = 0.0f;
    float my_sq = 0.0f;

    // Only threads 0~191 need to load data
    if (tid < 192) {
        int offset = token_id * EMBED_DIM + tid * 4;
        val = vload4(0, &input[offset]);

        // Sum components
        my_sum = val.x + val.y + val.z + val.w;
        // Sum squares
        my_sq = val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }

    s_sum[tid] = my_sum;
    s_sq[tid] = my_sq;
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. Parallel Reduction (Tree-based)
    // 256 -> 128 -> 64 -> ... -> 1
#pragma unroll
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sq[tid] += s_sq[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 3. Calculate Mean & Var (Thread 0 only)
    if (tid == 0) {
        float total_sum = s_sum[0];
        float total_sq = s_sq[0];

        s_mean = total_sum / (float)EMBED_DIM;
        float var = total_sq / (float)EMBED_DIM - s_mean * s_mean;
        s_invstd = rsqrt(var + EPS);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 4. Normalize & Store
    if (tid < 192) {
        // Load Gamma/Beta (Weight/Bias)
        float4 gamma = vload4(tid, weight);
        float4 beta = vload4(tid, bias);

        // Broadcast mean/invstd
        float4 res = (val - s_mean) * s_invstd * gamma + beta;

        int offset = token_id * EMBED_DIM + tid * 4;
        vstore4(res, 0, &output[offset]);
    }
}

// 4. Linear Layer (Matrix Multiplication)
 //Global Size: (tokens, out_features)
// [Optimized Standard Kernel] 일반 Linear Layer
// MLP에서 검증된 Tiling + Padding [16][17] + Unrolling 적용
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

    const int TSz = 16;

    // ★ Padding [16][17] 적용 (Bank Conflict 제거)
    __local float As[16][17];
    __local float Bs[16][17];

    float sum = 0.0f;
    int num_tiles = (in_features + TSz - 1) / TSz;

#pragma unroll
    for (int t = 0; t < num_tiles; ++t) {
        int k_base = t * TSz;
        int a_col = k_base + local_col;
        int b_row = k_base + local_row;

        if (row < tokens && a_col < in_features)
            As[local_row][local_col] = input[row * in_features + a_col];
        else
            As[local_row][local_col] = 0.0f;

        if (col < out_features && b_row < in_features)
            Bs[local_row][local_col] = weight[col * in_features + b_row];
        else
            Bs[local_row][local_col] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        // ★ Loop Unrolling
#pragma unroll
        for (int k = 0; k < TSz; ++k) {
            sum += As[local_row][k] * Bs[k][local_col];
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
// [Optimized Tiled Kernel 3] Attention Score (Q * K^T)
// Global Size: (total_tokens, total_tokens, num_heads) -> (197, 197, 12)
// j(Key Token), i(Query Token), h(Head) 순서
// [FIXED] Attention Score Kernel
// Q * K^T 연산을 위해 K 로딩 시 Transpose 수행
__kernel void attn_score_kernel(__global const float* qkv,
    __global float* scores,
    int total_tokens,   // 197
    int batch_size)
{
    // 1. 인덱스 계산
    int j = get_global_id(0); // key token idx (0..T-1) -> padded
    int i = get_global_id(1); // query token idx (0..T-1) -> padded
    int bh = get_global_id(2);

    int local_j = get_local_id(0);
    int local_i = get_local_id(1);

    int b = bh / NUM_HEADS;
    int h = bh % NUM_HEADS;

    // [수정 1] 여기서 return 해버리면 Barrier가 깨집니다! 삭제하세요.
    // if (b >= batch_size || h >= NUM_HEADS || ... return; ) -> 삭제

    // 유효성 플래그 미리 계산
    bool valid_i = (i < total_tokens) && (b < batch_size);
    bool valid_j = (j < total_tokens) && (b < batch_size);

    const int TSz = 16;
    __local float As[16][17]; // Bank Conflict 방지 패딩
    __local float Bs[16][17];

    int stride = 3 * EMBED_DIM;
    int q_offset_base = h * HEAD_DIM;
    int k_offset_base = EMBED_DIM + h * HEAD_DIM;

    float sum = 0.0f;
    int num_tiles = HEAD_DIM / TSz;

    for (int t = 0; t < num_tiles; ++t) {
        int d_base = t * TSz;

        // [수정 2] 로딩할 때 유효 범위 체크 (범위 밖이면 0.0으로 채움)

        // Q Load: Row 'i'가 유효해야 함
        // As[local_i][local_j] -> Row: i, Col: d (local_j가 dim 역할)
        if (valid_i) {
            // Query는 Token Index 'i'를 따라감
            int token_q = b * total_tokens + i;
            As[local_i][local_j] = qkv[token_q * stride + q_offset_base + (d_base + local_j)];
        }
        else {
            As[local_i][local_j] = 0.0f;
        }

        // K Load (Transposed): Row 'j'가 유효해야 함
        // Bs[local_i][local_j] -> Row: d (local_i가 dim 역할), Col: j
        if (valid_j) {
            // Key는 Token Index 'j'를 따라감
            int token_k = b * total_tokens + j;
            Bs[local_i][local_j] = qkv[token_k * stride + k_offset_base + (d_base + local_i)];
        }
        else {
            Bs[local_i][local_j] = 0.0f;
        }

        // 모든 스레드가 로딩을 마칠 때까지 대기
        barrier(CLK_LOCAL_MEM_FENCE);

        // 연산 (범위 밖 스레드도 계산은 참여하되, 나중에 버림)
#pragma unroll
        for (int k = 0; k < TSz; ++k) {
            sum += As[local_i][k] * Bs[k][local_j];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // [수정 3] 저장할 때만 유효성 체크해서 쓰기
    if (valid_i && valid_j) {
        int out_idx = (((b * NUM_HEADS + h) * total_tokens + i) * total_tokens + j);
        scores[out_idx] = sum / sqrt((float)HEAD_DIM);
    }
}
// [Optimized Tiled Kernel 4] Attention Value (Scores * V)
// Global Size: (head_dim, total_tokens, num_heads) -> (64, 197, 12)
// d(Feature), i(Query Token), h(Head) 순서
__kernel void attn_value_kernel(__global const float* scores,
    __global const float* qkv,
    __global float* attn_output,
    int total_tokens,   // per img (197)
    int batch_size)
{
    // 1. 인덱스 확보
    int d = get_global_id(0); // Head Dim (0..63) -> padded to 64 or 80
    int i = get_global_id(1); // Token idx (0..196) -> padded to 208
    int bh = get_global_id(2);

    int local_d = get_local_id(0);
    int local_i = get_local_id(1);

    int b = bh / NUM_HEADS;
    int h = bh % NUM_HEADS;

    // [수정 1] Return 삭제! (여기서 나가면 Barrier에서 멈춤)
    // if (b >= batch_size ... return;) -> 삭제

    // 2. 내 스레드가 유효한지 확인하는 플래그 생성
    bool valid_batch = (b < batch_size && h < NUM_HEADS);
    bool valid_i = (i < total_tokens) && valid_batch;     // 행(Query/Output) 유효성
    bool valid_d = (d < HEAD_DIM) && valid_batch;         // 열(HeadDim) 유효성

    const int TSz = 16;
    __local float As[16][17]; // Bank Conflict 방지
    __local float Bs[16][17];

    int stride = 3 * EMBED_DIM;
    int v_offset_base = 2 * EMBED_DIM + h * HEAD_DIM;

    float sum = 0.0f;
    // total_tokens가 197이면 16으로 나누어 떨어지지 않으므로 올림 나눗셈 필요
    int num_tiles = (total_tokens + TSz - 1) / TSz;

    for (int t = 0; t < num_tiles; ++t) {
        int j_base = t * TSz;

        // 타일 내부 좌표에 해당하는 실제 j(Key/Value Token Index) 계산
        int col_j_for_A = j_base + local_d; // As 로딩용 j (Column)
        int row_j_for_B = j_base + local_i; // Bs 로딩용 j (Row)

        // [수정 2] As (Scores) 로딩: A[i][j]
        // 조건: 내 행(i)이 유효하고 & 로딩하려는 열(j)이 유효해야 함
        if (valid_i && (col_j_for_A < total_tokens)) {
            // score_idx = (b, h, i, j)
            int score_idx = (((b * NUM_HEADS + h) * total_tokens + i) * total_tokens + col_j_for_A);
            As[local_i][local_d] = scores[score_idx];
        }
        else {
            As[local_i][local_d] = 0.0f; // 패딩 영역 0으로 채움
        }

        // [수정 3] Bs (Values) 로딩: B[j][d]
        // 조건: 로딩하려는 행(j)이 유효하고 & 내 열(d)이 유효해야 함
        if ((row_j_for_B < total_tokens) && valid_d) {
            int token_v = b * total_tokens + row_j_for_B;
            int v_idx = token_v * stride + v_offset_base + d;
            Bs[local_i][local_d] = qkv[v_idx];
        }
        else {
            Bs[local_i][local_d] = 0.0f;
        }

        // Barrier: 모든 스레드가 로딩을 마칠 때까지 대기
        barrier(CLK_LOCAL_MEM_FENCE);

        // 연산 (패딩 스레드도 계산에는 참여하지만 0을 더하므로 영향 없음)
#pragma unroll
        for (int k = 0; k < TSz; ++k) {
            sum += As[local_i][k] * Bs[k][local_d];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // [수정 4] 최종 저장: 진짜 유효한 스레드만 저장
    if (valid_i && valid_d) {
        int token_o = b * total_tokens + i;
        int out_idx = token_o * EMBED_DIM + h * HEAD_DIM + d;
        attn_output[out_idx] = sum;
    }
}



// 8. Softmax (Applied per row in scores)
// Global Size: (heads, tokens)
// 각 스레드가 하나의 행(row)을 맡아서 Softmax 처리
__kernel void softmax_kernel(__global float* scores, int total_tokens)
{
    int tid = get_local_id(0);
    int group_id = get_group_id(0);
    int row_offset = group_id * total_tokens;

    __local float s_data[256];
    __local float s_max;
    __local float s_sum;

    // 1. Find Max 
    float my_val = -1e30f;
    if (tid < total_tokens) {
        my_val = scores[row_offset + tid];
    }
    s_data[tid] = my_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_data[tid + s] > s_data[tid]) {
                s_data[tid] = s_data[tid + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) s_max = s_data[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. Calculate Exp & Sum (native_exp -> exp)
    float my_exp = 0.0f;
    if (tid < total_tokens) {
        float val = scores[row_offset + tid];
        // [Fix] native_exp는 오차가 큽니다. exp 사용.
        my_exp = exp(val - s_max);
        scores[row_offset + tid] = my_exp;
    }

    s_data[tid] = my_exp;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) s_sum = s_data[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3. Normalize (native_recip -> 나눗셈 연산)
    if (tid < total_tokens) {
        // [Fix] 정밀도를 위해 직접 나눗셈
        scores[row_offset + tid] /= s_sum;
    }
}

// [Optimized Tiled Kernel 1] Linear + GELU
// 1. Tiling (Local Memory) 복구
// 2. Memory Padding [16][17] 적용 -> Bank Conflict 제거
// 3. Loop Unrolling 적용
__kernel void linear_gelu_kernel(__global const float* input,
    __global float* output,
    __global const float* weight,
    __constant const float* bias,
    int in_features,
    int out_features,
    int tokens)
{
    // 인덱스
    int row = get_global_id(0);
    int col = get_global_id(1);
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);

    const int TSz = 16;

    // ★ 핵심 최적화: Padding 적용 [16][17] ★
    // 열(Column)을 하나 늘려 메모리 충돌 방지
    __local float As[16][17];
    __local float Bs[16][17];

    float sum = 0.0f;
    int num_tiles = (in_features + TSz - 1) / TSz;

#pragma unroll
    for (int t = 0; t < num_tiles; ++t) {
        int k_base = t * TSz;
        int a_col = k_base + local_col;
        int b_row = k_base + local_row;

        // Load Global -> Local
        if (row < tokens && a_col < in_features)
            As[local_row][local_col] = input[row * in_features + a_col];
        else
            As[local_row][local_col] = 0.0f;

        if (col < out_features && b_row < in_features)
            Bs[local_row][local_col] = weight[col * in_features + b_row];
        else
            Bs[local_row][local_col] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute
        // ★ 핵심 최적화: Loop Unrolling ★
        // 16번의 반복을 컴파일러가 최적화하도록 지시
#pragma unroll
        for (int k = 0; k < TSz; ++k) {
            sum += As[local_row][k] * Bs[k][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Result + GELU
    if (row < tokens && col < out_features) {
        float val = sum + bias[col];
        // GELU
        output[row * out_features + col] = 0.5f * val * (1.0f + erf(val * 0.70710678f));
    }
}

// [Optimized Tiled Kernel 2] Linear + Add
__kernel void linear_add_kernel(__global const float* restrict input,
    __global float* restrict output,
    __global const float* restrict weight,
    __constant const float* restrict bias,
    int in_features,
    int out_features,
    int tokens,
    __global const float* restrict prev_state) // Skip Connection
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);

    const int TSz = 16;
    __local float As[16][17];
    __local float Bs[16][17];

    float sum = 0.0f;
    int num_tiles = (in_features + TSz - 1) / TSz;

#pragma unroll
    for (int t = 0; t < num_tiles; ++t) {
        int k_base = t * TSz;
        int a_col = k_base + local_col;
        int b_row = k_base + local_row;

        if (row < tokens && a_col < in_features)
            As[local_row][local_col] = input[row * in_features + a_col];
        else
            As[local_row][local_col] = 0.0f;

        if (col < out_features && b_row < in_features)
            Bs[local_row][local_col] = weight[col * in_features + b_row];
        else
            Bs[local_row][local_col] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
        for (int k = 0; k < TSz; ++k) {
            sum += As[local_row][k] * Bs[k][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < tokens && col < out_features) {
        float val = sum + bias[col];
        // [Note] restrict를 썼으므로 컴파일러가 output과 prev_state를 
        // 겹치지 않는 것으로 가정하고 안전하게 로드/스토어 순서를 잡거나, 
        // 하드웨어가 읽기-수정-쓰기를 정확히 처리합니다.
        output[row * out_features + col] = val + prev_state[row * out_features + col];
    }
}

__kernel void gather_cls_kernel(__global const float* tokens,
    __global float* cls_out,
    int tokens_per_img,
    int batch_size)
{
    int b = get_global_id(0); // 0..B-1
    int d = get_global_id(1); // 0..dim-1

    if (b >= batch_size || d >= EMBED_DIM) return;

    int cls_token_idx = b * tokens_per_img; // 각 이미지의 0번 토큰이 CLS
    int src_idx = cls_token_idx * EMBED_DIM + d;
    int dst_idx = b * EMBED_DIM + d;

    cls_out[dst_idx] = tokens[src_idx];
}