#include <stdio.h>
#include <stdlib.h>
#include <string.h> // memcpy 사용을 위해 필수
#include <math.h>   // exp, sqrt
#include <time.h>
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

// [변경] 배치 사이즈 정의
#define BATCH_SIZE 64

// Helper: Linear Args 설정
void set_linear_args(cl_kernel kernel,
    cl_mem in,
    cl_mem out,
    cl_mem w,
    cl_mem b,
    cl_mem residual,   // 새 인자
    int in_f,
    int out_f,
    int total_tokens,
    int add_residual)  // 새 인자
{
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &out);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &w);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &b);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &residual);
    clSetKernelArg(kernel, 5, sizeof(int), &in_f);
    clSetKernelArg(kernel, 6, sizeof(int), &out_f);
    clSetKernelArg(kernel, 7, sizeof(int), &total_tokens);
    clSetKernelArg(kernel, 8, sizeof(int), &add_residual);
}



void ViT_opencl(ImageData* image, cl_mem* d_networks, float** probabilities,
    cl_context context, cl_command_queue queue, cl_program program, cl_device_id device)
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
    // [추가] CLS 토큰을 모으는 커널
    cl_kernel k_gather = clCreateKernel(program, "gather_cls_kernel", &err);

    // 2. 크기 계산
    int num_patches = (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE); // 196
    int tokens_per_img = num_patches + 1;   // 197
    int dim = EMBED_DIM;                    // 768
    int hidden_dim = dim * MLP_RATIO;       // 3072

    // 전체 배치 토큰 수
    int total_batch_tokens = tokens_per_img * BATCH_SIZE;

    // 3. 메모리 할당 크기
    size_t size_input_img = sizeof(float) * 3 * IMG_SIZE * IMG_SIZE * BATCH_SIZE;
    size_t size_token_dim = sizeof(float) * total_batch_tokens * dim;
    size_t size_token_hidden = sizeof(float) * total_batch_tokens * hidden_dim;
    size_t size_scores = sizeof(float) * NUM_HEADS * tokens_per_img * tokens_per_img * BATCH_SIZE;
    size_t size_cls_out = sizeof(float) * NUM_CLASSES * BATCH_SIZE;

    // [추가] Gather된 CLS 토큰만 담을 버퍼 크기 (Batch * 768)
    size_t size_gathered_cls = sizeof(float) * BATCH_SIZE * dim;

    size_t local_linear[2] = { 16, 16 };
    size_t global_linear[2];

    // 4. 버퍼 생성
    cl_mem d_input_img = clCreateBuffer(context, CL_MEM_READ_ONLY, size_input_img, NULL, &err);
    cl_mem d_buf1 = clCreateBuffer(context, CL_MEM_READ_WRITE, size_token_hidden, NULL, &err);
    cl_mem d_buf2 = clCreateBuffer(context, CL_MEM_READ_WRITE, size_token_hidden, NULL, &err);
    cl_mem d_residual = clCreateBuffer(context, CL_MEM_READ_WRITE, size_token_dim, NULL, &err);
    cl_mem d_intermediate = clCreateBuffer(context, CL_MEM_READ_WRITE, size_token_hidden, NULL, &err);
    cl_mem d_scores = clCreateBuffer(context, CL_MEM_READ_WRITE, size_scores, NULL, &err);
    cl_mem d_cls_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_cls_out, NULL, &err);
    // [추가] CLS Gather Buffer
    cl_mem d_cls_gathered = clCreateBuffer(context, CL_MEM_READ_WRITE, size_gathered_cls, NULL, &err);

    // 배치 데이터를 담을 임시 호스트 메모리
    float* h_batch_img = (float*)malloc(size_input_img); // 입력용
    float* h_result_buf = (float*)malloc(size_cls_out);  // 출력용 (Batch * 1000)

    // 5. Inference Loop
    for (int i = 0; i < image->n; i += BATCH_SIZE) {

        int current_batch = (i + BATCH_SIZE > image->n) ? (image->n - i) : BATCH_SIZE;

        // (A) Host -> Device 복사 (Batch Packing)
        // [수정] memcpy를 사용하여 이미지 데이터를 연속된 메모리로 패킹
        size_t one_img_bytes = sizeof(float) * 3 * IMG_SIZE * IMG_SIZE;

        for (int b = 0; b < current_batch; b++) {
            float* dst = h_batch_img + (b * 3 * IMG_SIZE * IMG_SIZE);
            const float* src = image[i + b].data;
            memcpy(dst, src, one_img_bytes);
        }

        clEnqueueWriteBuffer(queue, d_input_img, CL_TRUE, 0,
            one_img_bytes * current_batch, h_batch_img, 0, NULL, NULL);

        // (B) Patch Embedding
        clSetKernelArg(k_conv2d, 0, sizeof(cl_mem), &d_input_img);
        clSetKernelArg(k_conv2d, 1, sizeof(cl_mem), &d_buf2);
        clSetKernelArg(k_conv2d, 2, sizeof(cl_mem), &d_networks[1]);
        clSetKernelArg(k_conv2d, 3, sizeof(cl_mem), &d_networks[2]);
        size_t gws_conv[1] = { (size_t)current_batch * num_patches * dim };
        clEnqueueNDRangeKernel(queue, k_conv2d, 1, NULL, gws_conv, NULL, 0, NULL, NULL);

        // (C) Prep (CLS + Pos)
        clSetKernelArg(k_prep, 0, sizeof(cl_mem), &d_buf2);
        clSetKernelArg(k_prep, 1, sizeof(cl_mem), &d_buf1);
        clSetKernelArg(k_prep, 2, sizeof(cl_mem), &d_networks[0]);
        clSetKernelArg(k_prep, 3, sizeof(cl_mem), &d_networks[3]);
        size_t gws_prep[1] = { (size_t)current_batch * tokens_per_img * dim };
        clEnqueueNDRangeKernel(queue, k_prep, 1, NULL, gws_prep, NULL, 0, NULL, NULL);

        // (D) Encoder Layers
        int net_idx = 4;
        int tokens_in_pass = tokens_per_img * current_batch;

        for (int l = 0; l < 12; l++) {
            // LN 1
            clSetKernelArg(k_ln, 0, sizeof(cl_mem), &d_buf1);
            clSetKernelArg(k_ln, 1, sizeof(cl_mem), &d_buf2);
            clSetKernelArg(k_ln, 2, sizeof(cl_mem), &d_networks[net_idx + 0]); // ln1 gamma
            clSetKernelArg(k_ln, 3, sizeof(cl_mem), &d_networks[net_idx + 1]); // ln1 beta
            size_t gws_ln[1] = { (size_t)tokens_in_pass };
            clEnqueueNDRangeKernel(queue, k_ln, 1, NULL, gws_ln, NULL, 0, NULL, NULL);

            // MHA - Linear 1 (QKV)
            set_linear_args(k_linear,
                d_buf2, d_intermediate,
                d_networks[net_idx + 2], d_networks[net_idx + 3],
                (cl_mem)NULL,          // residual 없음
                dim, dim * 3, tokens_in_pass,
                0);                    // add_residual = 0

            global_linear[0] = ((tokens_in_pass + 15) / 16) * 16;
            global_linear[1] = ((dim * 3 + 15) / 16) * 16;
            clEnqueueNDRangeKernel(queue, k_linear, 2, NULL,
                global_linear, local_linear, 0, NULL, NULL);

            // MHA - Score
            clSetKernelArg(k_attn_score, 0, sizeof(cl_mem), &d_intermediate);
            clSetKernelArg(k_attn_score, 1, sizeof(cl_mem), &d_scores);
            clSetKernelArg(k_attn_score, 2, sizeof(int), &tokens_per_img);
            size_t gws_score[3] = { (size_t)current_batch * NUM_HEADS,
                                    (size_t)tokens_per_img,
                                    (size_t)tokens_per_img };
            clEnqueueNDRangeKernel(queue, k_attn_score, 3, NULL,
                gws_score, NULL, 0, NULL, NULL);

            // MHA - Softmax
            clSetKernelArg(k_softmax, 0, sizeof(cl_mem), &d_scores);
            clSetKernelArg(k_softmax, 1, sizeof(int), &tokens_per_img);
            size_t gws_softmax[2] = { (size_t)current_batch * NUM_HEADS,
                                      (size_t)tokens_per_img };
            clEnqueueNDRangeKernel(queue, k_softmax, 2, NULL,
                gws_softmax, NULL, 0, NULL, NULL);

            // MHA - Values
            clSetKernelArg(k_attn_val, 0, sizeof(cl_mem), &d_scores);
            clSetKernelArg(k_attn_val, 1, sizeof(cl_mem), &d_intermediate);
            clSetKernelArg(k_attn_val, 2, sizeof(cl_mem), &d_buf2);
            clSetKernelArg(k_attn_val, 3, sizeof(int), &tokens_per_img);
            size_t gws_val[3] = { (size_t)current_batch * NUM_HEADS,
                                  (size_t)tokens_per_img,
                                  (size_t)HEAD_DIM / 4 };
            clEnqueueNDRangeKernel(queue, k_attn_val, 3, NULL,
                gws_val, NULL, 0, NULL, NULL);

            // MHA - Final Linear + Add 1 (fused)
            // input  : d_buf2  (attn output)
            // output : d_buf1  (residual 적용된 결과)
            // residual: 이전 블록 입력이 들어 있는 d_buf1
            set_linear_args(k_linear,
                d_buf2, d_buf1,
                d_networks[net_idx + 4], d_networks[net_idx + 5],
                d_buf1,
                dim, dim, tokens_in_pass,
                1);   // add_residual = 1

            global_linear[0] = ((tokens_in_pass + 15) / 16) * 16;
            global_linear[1] = ((dim + 15) / 16) * 16;
            clEnqueueNDRangeKernel(queue, k_linear, 2, NULL,
                global_linear, local_linear, 0, NULL, NULL);

            // LN 2
            clSetKernelArg(k_ln, 0, sizeof(cl_mem), &d_buf1);
            clSetKernelArg(k_ln, 1, sizeof(cl_mem), &d_buf2);
            clSetKernelArg(k_ln, 2, sizeof(cl_mem), &d_networks[net_idx + 6]); // ln2 gamma
            clSetKernelArg(k_ln, 3, sizeof(cl_mem), &d_networks[net_idx + 7]); // ln2 beta
            clEnqueueNDRangeKernel(queue, k_ln, 1, NULL, gws_ln, NULL, 0, NULL, NULL);

            // MLP FC1
            set_linear_args(k_linear,
                d_buf2, d_intermediate,
                d_networks[net_idx + 8], d_networks[net_idx + 9],
                (cl_mem)NULL,
                dim, hidden_dim, tokens_in_pass,
                0);   // add_residual = 0

            global_linear[0] = ((tokens_in_pass + 15) / 16) * 16;
            global_linear[1] = ((hidden_dim + 15) / 16) * 16;
            clEnqueueNDRangeKernel(queue, k_linear, 2, NULL,
                global_linear, local_linear, 0, NULL, NULL);

            // GELU (on d_intermediate)
            clSetKernelArg(k_gelu, 0, sizeof(cl_mem), &d_intermediate);
            size_t gws_gelu[1] = { (size_t)tokens_in_pass * hidden_dim };
            clEnqueueNDRangeKernel(queue, k_gelu, 1, NULL,
                gws_gelu, NULL, 0, NULL, NULL);

            // MLP FC2 + Add 2 (fused)
            set_linear_args(k_linear,
                d_intermediate, d_buf1,
                d_networks[net_idx + 10], d_networks[net_idx + 11],
                d_buf1,
                hidden_dim, dim, tokens_in_pass,
                1);   // add_residual = 1

            global_linear[0] = ((tokens_in_pass + 15) / 16) * 16;
            global_linear[1] = ((dim + 15) / 16) * 16;
            clEnqueueNDRangeKernel(queue, k_linear, 2, NULL,
                global_linear, local_linear, 0, NULL, NULL);

            net_idx += 12;
        }


        // (E) Final LN
        clSetKernelArg(k_ln, 0, sizeof(cl_mem), &d_buf1);
        clSetKernelArg(k_ln, 1, sizeof(cl_mem), &d_buf2);
        clSetKernelArg(k_ln, 2, sizeof(cl_mem), &d_networks[148]);
        clSetKernelArg(k_ln, 3, sizeof(cl_mem), &d_networks[149]);
        size_t gws_ln_final[1] = { (size_t)tokens_in_pass };
        clEnqueueNDRangeKernel(queue, k_ln, 1, NULL, gws_ln_final, NULL, 0, NULL, NULL);

        // =================================================================
        // (F) Gather CLS Token & Classifier Head
        // =================================================================

        // 1. [Gather] 흩어져 있는 CLS 토큰을 d_cls_gathered로 모음
        clSetKernelArg(k_gather, 0, sizeof(cl_mem), &d_buf2);          // Input: All tokens
        clSetKernelArg(k_gather, 1, sizeof(cl_mem), &d_cls_gathered);  // Output: CLS tokens only
        clSetKernelArg(k_gather, 2, sizeof(int), &tokens_per_img);     // 197

        // GWS = Batch * 768
        size_t gws_gather[1] = { (size_t)current_batch * dim };
        clEnqueueNDRangeKernel(queue, k_gather, 1, NULL, gws_gather, NULL, 0, NULL, NULL);

        // 2. [Classifier] 모아진 CLS 토큰을 입력으로 사용
        // Input: d_cls_gathered (연속된 [Batch x 768])
        // Tokens: current_batch (Batch x 1)
        set_linear_args(k_linear,
                d_cls_gathered, d_cls_out,
                d_networks[150], d_networks[151],
                (cl_mem)NULL,
                dim, NUM_CLASSES, current_batch,
                0);



        global_linear[0] = ((current_batch + 15) / 16) * 16;
        global_linear[1] = ((NUM_CLASSES + 15) / 16) * 16;
        clEnqueueNDRangeKernel(queue, k_linear, 2, NULL, global_linear, local_linear, 0, NULL, NULL);

        // (G) Read Result
        // d_cls_out에 결과가 연속적으로 [Batch x 1000] 들어있으므로 한 번에 읽음
        clEnqueueReadBuffer(queue, d_cls_out, CL_TRUE, 0,
            sizeof(float) * current_batch * NUM_CLASSES, h_result_buf, 0, NULL, NULL);

        // Post Processing (Softmax)
        for (int b = 0; b < current_batch; b++) {
            // b번째 결과 포인터
            float* logits = h_result_buf + (b * NUM_CLASSES);

            // Softmax
            float max_val = logits[0];
            for (int k = 1; k < NUM_CLASSES; k++) if (logits[k] > max_val) max_val = logits[k];

            float sum = 0;
            for (int k = 0; k < NUM_CLASSES; k++) {
                probabilities[i + b][k] = exp(logits[k] - max_val);
                sum += probabilities[i + b][k];
            }
            for (int k = 0; k < NUM_CLASSES; k++) probabilities[i + b][k] /= sum;
        }
    }

    // Cleanup
    free(h_batch_img);
    free(h_result_buf);

    clReleaseMemObject(d_input_img);
    clReleaseMemObject(d_buf1);
    clReleaseMemObject(d_buf2);
    clReleaseMemObject(d_residual);
    clReleaseMemObject(d_intermediate);
    clReleaseMemObject(d_scores);
    clReleaseMemObject(d_cls_out);
    clReleaseMemObject(d_cls_gathered); // [추가]

    clReleaseKernel(k_conv2d);
    clReleaseKernel(k_prep);
    clReleaseKernel(k_ln);
    clReleaseKernel(k_linear);
    clReleaseKernel(k_add);
    clReleaseKernel(k_gelu);
    clReleaseKernel(k_attn_score);
    clReleaseKernel(k_softmax);
    clReleaseKernel(k_attn_val);
    clReleaseKernel(k_gather); // [추가]
}