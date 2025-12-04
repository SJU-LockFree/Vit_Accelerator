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
    __constant const float* bias)
{
    int idx = get_global_id(0);
    int output_size = IMG_SIZE / PATCH_SIZE; // 14
    int total_patches = output_size * output_size; // 196

    // 범위 체크
    if (idx >= total_patches * EMBED_DIM) return;

    // 인덱스 분해
    int oc = idx / total_patches;         // Output Channel (Filter Index)
    int patch_idx = idx % total_patches;  // Patch Index (Spatial Index)

    // 현재 처리할 패치의 이미지 상 좌표 (좌상단)
    int oh = patch_idx / output_size;     // Patch Row
    int ow = patch_idx % output_size;     // Patch Col

    int start_y = oh * PATCH_SIZE;
    int start_x = ow * PATCH_SIZE;

    // Bias 로드
    float sum = bias[oc];

    // Weight 오프셋 미리 계산 (Out Channel 차원 이동)
    // Weight Shape: [Out_Ch, In_Ch, KH, KW]
    int weight_oc_offset = oc * IN_CHANS * PATCH_SIZE * PATCH_SIZE;

    // Loop Unrolling: 입력 채널(3)은 반복 횟수가 적으므로 루프 오버헤드 최소화
#pragma unroll 
    for (int ic = 0; ic < IN_CHANS; ++ic) {

        // Input Image Offset (Channel 차원 이동)
        // Input Shape: [In_Ch, Height, Width]
        int input_ic_offset = ic * IMG_SIZE * IMG_SIZE;

        // Weight Offset (Input Channel 차원 이동)
        int weight_ic_offset = weight_oc_offset + (ic * PATCH_SIZE * PATCH_SIZE);

        // 커널 높이(16) 반복
#pragma unroll 
        for (int kh = 0; kh < PATCH_SIZE; ++kh) {

            // 현재 행(Row)의 시작 포인터 계산
            int input_row_idx = input_ic_offset + (start_y + kh) * IMG_SIZE + start_x;
            int weight_row_idx = weight_ic_offset + kh * PATCH_SIZE;

            // ★ 핵심 최적화: 가로 16픽셀을 float4 x 4번으로 처리 ★
            // PATCH_SIZE(16) / 4 = 4 iterations
            // float4를 사용하면 128비트(16바이트)씩 한 번에 읽어옵니다.

            // Vector 0 (pixels 0~3)
            float4 in_val = vload4(0, &input[input_row_idx]);
            float4 w_val = vload4(0, &weight[weight_row_idx]);
            sum += dot(in_val, w_val);

            // Vector 1 (pixels 4~7)
            in_val = vload4(1, &input[input_row_idx]); // 오프셋 1 = 4 floats
            w_val = vload4(1, &weight[weight_row_idx]);
            sum += dot(in_val, w_val);

            // Vector 2 (pixels 8~11)
            in_val = vload4(2, &input[input_row_idx]);
            w_val = vload4(2, &weight[weight_row_idx]);
            sum += dot(in_val, w_val);

            // Vector 3 (pixels 12~15)
            in_val = vload4(3, &input[input_row_idx]);
            w_val = vload4(3, &weight[weight_row_idx]);
            sum += dot(in_val, w_val);
        }
    }

    // 결과 저장 (Flatten & Transpose: Patch -> Channel 순서)
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
    __constant const float* weight,     /* [ constant화 ] */
    __constant const float* bias)       /* [ constant화 ] */
{
    int t = get_global_id(0); // token index
    int total_tokens = (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE) + 1;
    if (t >= total_tokens) return;

    int offset = t * EMBED_DIM;

    // 1. [준비] float 포인터를 float4 포인터로 변환 (Type Casting)
    // 이제 배열 인덱스 1이 증가할 때마다 실제로는 float 4칸씩 이동합니다.
    __global const float4* in_vec = (__global const float4*)(input + offset);
    __global float4* out_vec = (__global float4*)(output + offset);
    __constant const float4* gamma_vec = (__constant const float4*)weight;
    __constant const float4* beta_vec = (__constant const float4*)bias;

    // 2. [Mean, Var 계산] 벡터 루프
    float4 v_sum = (float4)(0.0f);    // 4개 성분 0으로 초기화
    float4 v_sum_sq = (float4)(0.0f);

    // 루프 횟수가 1/4로 감소 (768 -> 192)
#pragma unroll 
    for (int i = 0; i < EMBED_DIM / 4; i++) {
        float4 val = in_vec[i];       // 4개 데이터 한 번에 로딩 (Load)
        v_sum += val;                 // 4개 덧셈 동시 수행 (SIMD)
        v_sum_sq += val * val;        // 4개 곱셈 -> 4개 덧셈
    }

    // [벡터 -> 스칼라 리덕션]
    // 4개로 쪼개져 있는 합계를 하나로 모음 (.x, .y, .z, .w)
    float sum = v_sum.x + v_sum.y + v_sum.z + v_sum.w;
    float sum_sq = v_sum_sq.x + v_sum_sq.y + v_sum_sq.z + v_sum_sq.w;

    float mean = sum / (float)EMBED_DIM;
    float var = sum_sq / (float)EMBED_DIM - mean * mean;
    float invstd = 1.0f / sqrt(var + EPS);

    // 3. [정규화] 벡터 루프
#pragma unroll
    for (int i = 0; i < EMBED_DIM / 4; i++) {
        float4 val = in_vec[i];
        float4 gamma = gamma_vec[i];
        float4 beta = beta_vec[i];

        // 스칼라(mean, invstd)와 벡터(val)의 연산은
        // OpenCL이 알아서 스칼라를 모든 벡터 성분에 적용(Broadcast)해줍니다.
        // val(4개) - mean(1개) -> 각 성분에서 mean을 뺌
        float4 res = (val - mean) * invstd * gamma + beta;

        out_vec[i] = res; // 4개 데이터 한 번에 저장 (Store)
    }
}

// 4. Linear Layer (Matrix Multiplication)
 //Global Size: (tokens, out_features)
__kernel void linear_kernel(__global const float* input,
    __global float* output,
    __global const float* weight,
    __constant const float* bias,    /* [ constant화 ] */
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

#pragma unroll                                 /* [ unloop 적용 ] */
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
    // 0.5 * x * (1 + erf(x / sqrt(2)))
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

    // [1] 누적 변수를 float4 벡터로 선언 (0.0으로 초기화)
    float4 score_vec = (float4)(0.0f);

    // [2] 루프를 4칸씩 점프하며 순회
    // #pragma unroll을 사용해 루프 오버헤드를 더 줄일 수 있음
#pragma unroll 4
    for (int d = 0; d < HEAD_DIM; d += 4) {
        // Q Base Index
        int q_base = i * stride + (h * HEAD_DIM + d);

        // K Base Index
        int k_base = j * stride + (EMBED_DIM + h * HEAD_DIM + d);

        // [3] vload4: 메모리에서 float 4개를 한 번에 읽어옴
        // &qkv[q_base] 주소부터 4개를 읽음
        float4 vec_q = vload4(0, &qkv[q_base]);
        float4 vec_k = vload4(0, &qkv[k_base]);

        // [4] 벡터 곱셈 & 누적
        // (x*x, y*y, z*z, w*w)가 동시에 계산되어 score_vec에 더해짐
        score_vec += vec_q * vec_k;
    }

    // [5] 수평 합산 (Horizontal Sum)
    // 벡터의 4개 성분(x, y, z, w)을 모두 더해 최종 스칼라 값 생성
    float final_score = score_vec.x + score_vec.y + score_vec.z + score_vec.w;

    scores[(h * total_tokens + i) * total_tokens + j] =
        final_score / sqrt((float)HEAD_DIM);
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
#pragma unroll
    for (int j = 1; j < total_tokens; j++) {
        float val = scores[row_offset + j];
        if (val > max_val) max_val = val;
    }

    // Exp & Sum
    float sum_exp = 0.0f;
#pragma unroll
    for (int j = 0; j < total_tokens; j++) {
        float val = exp(scores[row_offset + j] - max_val);
        scores[row_offset + j] = val;
        sum_exp += val;
    }

    // Normalize
    float inv_sum = 1.0f / sum_exp;
#pragma unroll
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
    int i = get_global_id(1); // query token (output row)
    int d_vec = get_global_id(2); // head dim / 4 (Vector index)

    int d = d_vec * 4; // 실제 시작 인덱스 (0, 4, 8...)

    // HEAD_DIM은 보통 64이므로 4의 배수라고 가정합니다.
    if (h >= NUM_HEADS || i >= total_tokens || d >= HEAD_DIM) return;

    int row_offset = (h * total_tokens + i) * total_tokens;
    int stride = 3 * EMBED_DIM; // Q, K, V가 합쳐진 stride

    // V 벡터의 시작 오프셋 (Q, K 다음)
    // V starts at: 2 * EMBED_DIM + h * HEAD_DIM
    int v_base_offset = 2 * EMBED_DIM + h * HEAD_DIM;

    // 누적 합을 위한 float4 벡터 초기화
    float4 sum_vec = (float4)(0.0f);

    // [Loop Unrolling] 컴파일러에게 루프 최적화 지시
#pragma unroll 4
    for (int j = 0; j < total_tokens; j++) {
        // 1. Score는 스칼라 값 (Scalar Load)
        // j번째 토큰에 대한 attention score (모든 d에 대해 동일)
        float s = scores[row_offset + j];

        // 2. V값은 4개를 한 번에 로딩 (Vector Load)
        // 로딩 위치: j번째 토큰의 V 벡터 중 d번째 요소부터 4개
        int v_idx = j * stride + v_base_offset + d;
        float4 v_val = vload4(0, &qkv[v_idx]);

        // 3. 연산 (Scalar * Vector)
        // s가 v_val의 x, y, z, w 성분에 각각 곱해짐
        sum_vec += s * v_val;
    }

    // 4. 최종 결과 저장 (Vector Store)
    // [tokens, EMBED_DIM] 구조
    // EMBED_DIM = NUM_HEADS * HEAD_DIM
    int out_idx = i * EMBED_DIM + h * HEAD_DIM + d;
    vstore4(sum_vec, 0, &attn_output[out_idx]);
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
__kernel void linear_add_kernel(__global const float* input,
    __global float* output,
    __global const float* weight,
    __constant const float* bias,
    int in_features,
    int out_features,
    int tokens,
    __global const float* prev_state) // Skip Connection
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);

    const int TSz = 16;

    // ★ Padding [16][17]
    __local float As[16][17];
    __local float Bs[16][17];

    float sum = 0.0f;
    int num_tiles = (in_features + TSz - 1) / TSz;

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

        // Compute
#pragma unroll
        for (int k = 0; k < TSz; ++k) {
            sum += As[local_row][k] * Bs[k][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Result + Add
    if (row < tokens && col < out_features) {
        float val = sum + bias[col];
        // Add Residual
        output[row * out_features + col] = val + prev_state[row * out_features + col];
    }
}