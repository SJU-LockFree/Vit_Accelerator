#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "Network.h"
#include "ViT_opencl.h"

// ViT Constants
#define IMG_SIZE 224
#define PATCH_SIZE 16
#define EMBED_DIM 768
#define MLP_RATIO 4
#define NUM_HEADS 12
#define HEAD_DIM 64
#define NUM_CLASSES 1000

// Helper to set standard linear args
void set_linear_args(cl_kernel kernel, cl_mem in, cl_mem out, cl_mem w, cl_mem b, int in_f, int out_f, int num_tokens) {
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &out);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &w);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &b);
    clSetKernelArg(kernel, 4, sizeof(int), &in_f);
    clSetKernelArg(kernel, 5, sizeof(int), &out_f);
    clSetKernelArg(kernel, 6, sizeof(int), &num_tokens);
}

void ViT_opencl(ImageData* image, cl_mem* d_networks, float** probabilities,
    cl_context context, cl_command_queue queue, cl_program program)
{
    cl_int err;

    // 1. 커널 생성
    cl_kernel k_conv2d = clCreateKernel(program, "conv2d_kernel", &err);
    cl_kernel k_prep = clCreateKernel(program, "prepare_input_kernel", &err);
    cl_kernel k_ln = clCreateKernel(program, "layer_norm_kernel", &err);
    cl_kernel k_linear = clCreateKernel(program, "linear_kernel", &err);
    cl_kernel k_add = clCreateKernel(program, "add_kernel", &err);
    cl_kernel k_gelu = clCreateKernel(program, "gelu_kernel", &err);
    cl_kernel k_attn_score = clCreateKernel(program, "attn_score_kernel", &err);
    cl_kernel k_softmax = clCreateKernel(program, "softmax_kernel", &err);
    cl_kernel k_attn_val = clCreateKernel(program, "attn_value_kernel", &err);

    // 2. 버퍼 크기 계산
    int num_patches = (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE); // 196
    int tokens = num_patches + 1; // 197
    int dim = EMBED_DIM;          // 768
    int hidden_dim = dim * MLP_RATIO; // 3072

    // 버퍼 사이즈 정의
    size_t size_token_dim = sizeof(float) * tokens * dim;           // 197 * 768
    size_t size_token_hidden = sizeof(float) * tokens * hidden_dim; // 197 * 3072 (가장 큼!)
    size_t size_scores = sizeof(float) * NUM_HEADS * tokens * tokens;

    size_t local_linear[2] = { 16, 16 };
    size_t global_linear[2];

    // 입력 이미지 버퍼
    cl_mem d_input_img = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 3 * IMG_SIZE * IMG_SIZE, NULL, &err);

    // 메인 버퍼 (Ping-Pong용)
    cl_mem d_buf1 = clCreateBuffer(context, CL_MEM_READ_WRITE, size_token_hidden, NULL, &err);
    cl_mem d_buf2 = clCreateBuffer(context, CL_MEM_READ_WRITE, size_token_hidden, NULL, &err);
    cl_mem d_residual = clCreateBuffer(context, CL_MEM_READ_WRITE, size_token_dim, NULL, &err);

    // ★수정됨★: QKV 뿐만 아니라 MLP의 Hidden Layer(4배)도 담을 수 있도록 가장 큰 크기로 할당
    cl_mem d_intermediate = clCreateBuffer(context, CL_MEM_READ_WRITE, size_token_hidden, NULL, &err);

    // Attention Score 버퍼
    cl_mem d_scores = clCreateBuffer(context, CL_MEM_READ_WRITE, size_scores, NULL, &err);

    // 최종 Output 버퍼
    cl_mem d_cls_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * NUM_CLASSES, NULL, &err);

    // 3. Inference Loop
    for (int img_idx = 0; img_idx < image->n; img_idx++) {

        // (A) 이미지 복사 Host -> GPU
        clEnqueueWriteBuffer(queue, d_input_img, CL_TRUE, 0, sizeof(float) * 3 * IMG_SIZE * IMG_SIZE,
            &image[img_idx].data[0], 0, NULL, NULL);

        // (B) Patch Embedding
        clSetKernelArg(k_conv2d, 0, sizeof(cl_mem), &d_input_img);
        clSetKernelArg(k_conv2d, 1, sizeof(cl_mem), &d_buf2);
        clSetKernelArg(k_conv2d, 2, sizeof(cl_mem), &d_networks[1]);
        clSetKernelArg(k_conv2d, 3, sizeof(cl_mem), &d_networks[2]);
        size_t gws_conv[1] = { num_patches * dim };
        clEnqueueNDRangeKernel(queue, k_conv2d, 1, NULL, gws_conv, NULL, 0, NULL, NULL);

        // (C) CLS Token + Pos Embedding
        clSetKernelArg(k_prep, 0, sizeof(cl_mem), &d_buf2);
        clSetKernelArg(k_prep, 1, sizeof(cl_mem), &d_buf1);
        clSetKernelArg(k_prep, 2, sizeof(cl_mem), &d_networks[0]);
        clSetKernelArg(k_prep, 3, sizeof(cl_mem), &d_networks[3]);
        size_t gws_prep[1] = { tokens * dim };
        clEnqueueNDRangeKernel(queue, k_prep, 1, NULL, gws_prep, NULL, 0, NULL, NULL);

        // (D) Encoder Layers
        int net_idx = 4;
        for (int i = 0; i < 12; i++) {
            // --- LN1 ---
            clSetKernelArg(k_ln, 0, sizeof(cl_mem), &d_buf1);
            clSetKernelArg(k_ln, 1, sizeof(cl_mem), &d_buf2);
            clSetKernelArg(k_ln, 2, sizeof(cl_mem), &d_networks[net_idx + 0]);
            clSetKernelArg(k_ln, 3, sizeof(cl_mem), &d_networks[net_idx + 1]);
            size_t gws_ln[1] = { tokens };
            clEnqueueNDRangeKernel(queue, k_ln, 1, NULL, gws_ln, NULL, 0, NULL, NULL);

            // --- Multi-Head Attention ---
            // 1. Linear Projection (Input: d_buf2 -> Output: d_intermediate)
            // d_intermediate will hold QKV (Size: tokens * 3 * dim) -> Safe!
            set_linear_args(k_linear, d_buf2, d_intermediate, d_networks[net_idx + 2], d_networks[net_idx + 3], dim, dim * 3, tokens);

            // Global Size 패딩: (tokens, 3*dim)을 16의 배수로 올림
            global_linear[0] = ((tokens + 15) / 16) * 16;
            global_linear[1] = ((dim * 3 + 15) / 16) * 16;


            clEnqueueNDRangeKernel(queue, k_linear, 2, NULL, global_linear, local_linear, 0, NULL, NULL);

            // 2. Calc Scores
            clSetKernelArg(k_attn_score, 0, sizeof(cl_mem), &d_intermediate);
            clSetKernelArg(k_attn_score, 1, sizeof(cl_mem), &d_scores);
            clSetKernelArg(k_attn_score, 2, sizeof(int), &tokens);
            size_t gws_score[3] = { NUM_HEADS, tokens, tokens };
            clEnqueueNDRangeKernel(queue, k_attn_score, 3, NULL, gws_score, NULL, 0, NULL, NULL);

            // 3. Softmax
            clSetKernelArg(k_softmax, 0, sizeof(cl_mem), &d_scores);
            clSetKernelArg(k_softmax, 1, sizeof(int), &tokens);
            size_t gws_softmax[2] = { NUM_HEADS, tokens };
            clEnqueueNDRangeKernel(queue, k_softmax, 2, NULL, gws_softmax, NULL, 0, NULL, NULL);

            // 4. Calc Values (Scores * V) -> Output: d_buf2
            clSetKernelArg(k_attn_val, 0, sizeof(cl_mem), &d_scores);
            clSetKernelArg(k_attn_val, 1, sizeof(cl_mem), &d_intermediate);
            clSetKernelArg(k_attn_val, 2, sizeof(cl_mem), &d_buf2);
            clSetKernelArg(k_attn_val, 3, sizeof(int), &tokens);
            size_t gws_val[3] = { NUM_HEADS, tokens, HEAD_DIM };
            clEnqueueNDRangeKernel(queue, k_attn_val, 3, NULL, gws_val, NULL, 0, NULL, NULL);

            // 5. Final Linear (Proj) -> Output: d_residual
            set_linear_args(k_linear, d_buf2, d_residual, d_networks[net_idx + 4], d_networks[net_idx + 5], dim, dim, tokens);

            global_linear[0] = ((tokens + 15) / 16) * 16;
            global_linear[1] = ((dim + 15) / 16) * 16;

            clEnqueueNDRangeKernel(queue, k_linear, 2, NULL, global_linear, local_linear, 0, NULL, NULL);

            // --- Residual Add 1 ---
            clSetKernelArg(k_add, 0, sizeof(cl_mem), &d_residual);
            clSetKernelArg(k_add, 1, sizeof(cl_mem), &d_buf1);
            size_t gws_add[1] = { tokens * dim };
            clEnqueueNDRangeKernel(queue, k_add, 1, NULL, gws_add, NULL, 0, NULL, NULL);

            // --- LN2 ---
            clSetKernelArg(k_ln, 0, sizeof(cl_mem), &d_buf1);
            clSetKernelArg(k_ln, 1, sizeof(cl_mem), &d_buf2);
            clSetKernelArg(k_ln, 2, sizeof(cl_mem), &d_networks[net_idx + 6]);
            clSetKernelArg(k_ln, 3, sizeof(cl_mem), &d_networks[net_idx + 7]);
            clEnqueueNDRangeKernel(queue, k_ln, 1, NULL, gws_ln, NULL, 0, NULL, NULL);

            // --- MLP ---
            // 1. FC1: d_buf2 -> d_intermediate
            // ★중요★: 여기서 d_intermediate는 Hidden Dim (4배) 크기의 데이터를 받습니다.
            // 이전 코드에서는 여기가 size_qkv(3배)여서 터졌던 것입니다.
            set_linear_args(k_linear, d_buf2, d_intermediate, d_networks[net_idx + 8], d_networks[net_idx + 9], dim, hidden_dim, tokens);

            global_linear[0] = ((tokens + 15) / 16) * 16;
            global_linear[1] = ((hidden_dim + 15) / 16) * 16;

            clEnqueueNDRangeKernel(queue, k_linear, 2, NULL, global_linear, local_linear, 0, NULL, NULL);

            // 2. GELU
            clSetKernelArg(k_gelu, 0, sizeof(cl_mem), &d_intermediate);
            size_t gws_gelu[1] = { tokens * hidden_dim };
            clEnqueueNDRangeKernel(queue, k_gelu, 1, NULL, gws_gelu, NULL, 0, NULL, NULL);

            // 3. FC2: d_intermediate -> d_residual
            set_linear_args(k_linear, d_intermediate, d_residual, d_networks[net_idx + 10], d_networks[net_idx + 11], hidden_dim, dim, tokens);

            global_linear[0] = ((tokens + 15) / 16) * 16;
            global_linear[1] = ((dim + 15) / 16) * 16;

            clEnqueueNDRangeKernel(queue, k_linear, 2, NULL, global_linear, local_linear, 0, NULL, NULL);

            // --- Residual Add 2 ---
            clSetKernelArg(k_add, 0, sizeof(cl_mem), &d_residual);
            clSetKernelArg(k_add, 1, sizeof(cl_mem), &d_buf1);
            clEnqueueNDRangeKernel(queue, k_add, 1, NULL, gws_add, NULL, 0, NULL, NULL);

            net_idx += 12;
        }

        // (E) Final Layer Norm
        clSetKernelArg(k_ln, 0, sizeof(cl_mem), &d_buf1);
        clSetKernelArg(k_ln, 1, sizeof(cl_mem), &d_buf2);
        clSetKernelArg(k_ln, 2, sizeof(cl_mem), &d_networks[148]);
        clSetKernelArg(k_ln, 3, sizeof(cl_mem), &d_networks[149]);
        size_t gws_ln[1] = { tokens };
        clEnqueueNDRangeKernel(queue, k_ln, 1, NULL, gws_ln, NULL, 0, NULL, NULL);

        // (F) Classifier Head
        set_linear_args(k_linear, d_buf2, d_cls_out, d_networks[150], d_networks[151], dim, NUM_CLASSES, 1);

        global_linear[0] = 16; // 1 -> 16 Padding
        global_linear[1] = ((NUM_CLASSES + 15) / 16) * 16;

        clEnqueueNDRangeKernel(queue, k_linear, 2, NULL, global_linear, local_linear, 0, NULL, NULL);

        // (G) Read Result
        clEnqueueReadBuffer(queue, d_cls_out, CL_TRUE, 0, sizeof(float) * NUM_CLASSES,
            probabilities[img_idx], 0, NULL, NULL);

        // Softmax (CPU 계산 - 정확도 유지용)
        float max_val = probabilities[img_idx][0];
        for (int k = 1; k < NUM_CLASSES; k++) if (probabilities[img_idx][k] > max_val) max_val = probabilities[img_idx][k];
        float sum_exp = 0.0f;
        for (int k = 0; k < NUM_CLASSES; k++) {
            probabilities[img_idx][k] = exp(probabilities[img_idx][k] - max_val);
            sum_exp += probabilities[img_idx][k];
        }
        for (int k = 0; k < NUM_CLASSES; k++) probabilities[img_idx][k] /= sum_exp;
    }

    // 4. Clean up
    clReleaseMemObject(d_input_img);
    clReleaseMemObject(d_buf1);
    clReleaseMemObject(d_buf2);
    clReleaseMemObject(d_residual);
    clReleaseMemObject(d_intermediate);
    clReleaseMemObject(d_scores);
    clReleaseMemObject(d_cls_out);

    clReleaseKernel(k_conv2d);
    clReleaseKernel(k_prep);
    clReleaseKernel(k_ln);
    clReleaseKernel(k_linear);
    clReleaseKernel(k_add);
    clReleaseKernel(k_gelu);
    clReleaseKernel(k_attn_score);
    clReleaseKernel(k_softmax);
    clReleaseKernel(k_attn_val);
}