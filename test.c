#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>
#include "Network.h"
#include "ViT_opencl.h"

// ==========================================
// [설정] 하이브리드 최적화 파라미터
// ==========================================
#define BATCH_SIZE 32   // 한 번에 32장 (Data Parallelism)
#define NUM_STREAMS 2   // 2개의 라인으로 파이프라이닝 (Task Parallelism)

// ViT Constants
#define IMG_SIZE 224
#define PATCH_SIZE 16
#define EMBED_DIM 768
#define MLP_RATIO 4
#define NUM_HEADS 12
#define HEAD_DIM 64
#define NUM_CLASSES 1000

// Helper: 커널 인자 세팅 (반복 사용을 위해 함수화)
void set_linear_args(cl_kernel kernel, cl_mem in, cl_mem out, cl_mem w, cl_mem b, int in_f, int out_f, int total_tokens) {
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &out);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &w);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &b);
    clSetKernelArg(kernel, 4, sizeof(int), &in_f);
    clSetKernelArg(kernel, 5, sizeof(int), &out_f);
    clSetKernelArg(kernel, 6, sizeof(int), &total_tokens);
}

void ViT_opencl(ImageData* image, cl_mem* d_networks, float** probabilities,
    cl_context context, cl_command_queue queue_dummy, cl_program program, cl_device_id device)
{
    cl_int err;

    // ------------------------------------------------------------
    // 1. 커널 생성 (공유 자원)
    // ------------------------------------------------------------
    cl_kernel k_conv2d = clCreateKernel(program, "conv2d_kernel", &err);
    cl_kernel k_prep = clCreateKernel(program, "prepare_input_kernel", &err);
    cl_kernel k_ln = clCreateKernel(program, "layer_norm_kernel", &err);
    cl_kernel k_linear = clCreateKernel(program, "linear_kernel", &err);
    cl_kernel k_add = clCreateKernel(program, "add_kernel", &err);
    cl_kernel k_gelu = clCreateKernel(program, "gelu_kernel", &err);
    cl_kernel k_attn_score = clCreateKernel(program, "attn_score_kernel", &err);
    cl_kernel k_softmax = clCreateKernel(program, "softmax_kernel", &err);
    cl_kernel k_attn_val = clCreateKernel(program, "attn_value_kernel", &err);
    cl_kernel k_gather = clCreateKernel(program, "gather_cls_kernel", &err);
    cl_kernel k_linear_gelu = clCreateKernel(program, "linear_gelu_kernel", &err);
    cl_kernel k_linear_add = clCreateKernel(program, "linear_add_kernel", &err);

    // ------------------------------------------------------------
    // 2. 크기 계산
    // ------------------------------------------------------------
    int num_patches = (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE);
    int tokens_per_img = num_patches + 1; // 197
    int dim = EMBED_DIM;
    int hidden_dim = dim * MLP_RATIO;

    // 배치 전체 토큰 수 (커널 GWS 계산용)
    int total_batch_tokens = tokens_per_img * BATCH_SIZE;

    // ------------------------------------------------------------
    // 3. 스트림별 리소스 할당 (이중 버퍼링)
    // ------------------------------------------------------------
    cl_command_queue queues[NUM_STREAMS];

    // GPU Buffers (Stream별 독립)
    cl_mem d_input_img[NUM_STREAMS];
    cl_mem d_buf1[NUM_STREAMS];
    cl_mem d_buf2[NUM_STREAMS];
    cl_mem d_residual[NUM_STREAMS];
    cl_mem d_intermediate[NUM_STREAMS];
    cl_mem d_scores[NUM_STREAMS];
    cl_mem d_cls_out[NUM_STREAMS];
    cl_mem d_cls_gathered[NUM_STREAMS];

    // Host Pinned Memory 대용 (비동기 전송 안전 보장용)
    float* h_stream_input[NUM_STREAMS];
    float* h_stream_output[NUM_STREAMS];

    size_t size_input_bytes = sizeof(float) * 3 * IMG_SIZE * IMG_SIZE * BATCH_SIZE;
    size_t size_output_bytes = sizeof(float) * NUM_CLASSES * BATCH_SIZE;

    // 리소스 생성 루프
    for (int s = 0; s < NUM_STREAMS; s++) {
        // (A) Command Queue (In-order)
        queues[s] = clCreateCommandQueueWithProperties(context, device, 0, &err);

        // (B) Device Buffers
        d_input_img[s] = clCreateBuffer(context, CL_MEM_READ_ONLY, size_input_bytes, NULL, &err);
        d_buf1[s] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * total_batch_tokens * hidden_dim, NULL, &err);
        d_buf2[s] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * total_batch_tokens * hidden_dim, NULL, &err);
        d_residual[s] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * total_batch_tokens * dim, NULL, &err);
        d_intermediate[s] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * total_batch_tokens * hidden_dim, NULL, &err);
        d_scores[s] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * NUM_HEADS * tokens_per_img * tokens_per_img * BATCH_SIZE, NULL, &err);
        d_cls_out[s] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_output_bytes, NULL, &err);
        d_cls_gathered[s] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * BATCH_SIZE * dim, NULL, &err);

        // (C) Host Temp Buffers (malloc)
        h_stream_input[s] = (float*)malloc(size_input_bytes);
        h_stream_output[s] = (float*)malloc(size_output_bytes);
    }

    size_t local_linear[2] = { 16, 16 };
    size_t global_linear[2];

    // ------------------------------------------------------------
    // 4. Inference Pipeline Loop
    // ------------------------------------------------------------
    // NUM_STREAMS만큼 더 돌아서 파이프라인을 비움 (Drain)
    int total_iterations = (image->n + BATCH_SIZE - 1) / BATCH_SIZE;

    for (int step = 0; step < total_iterations + NUM_STREAMS; step++) {

        int stream_id = step % NUM_STREAMS;
        cl_command_queue q = queues[stream_id];

        // [SYNC Point] 해당 스트림의 이전 작업이 끝날 때까지 대기
        // (즉, h_stream_output[stream_id]에 이전 결과가 다 들어왔는지 확인)
        clFinish(q);

        // =========================================================
        // Step 1: 이전 작업 결과 처리 (CPU Post-processing)
        // =========================================================
        // 방금 끝난 작업의 배치가 몇 번째였는지 역계산
        int completed_batch_idx = step - NUM_STREAMS;

        if (completed_batch_idx >= 0 && completed_batch_idx < total_iterations) {
            int img_start_idx = completed_batch_idx * BATCH_SIZE;
            int current_batch_size = (img_start_idx + BATCH_SIZE > image->n) ? (image->n - img_start_idx) : BATCH_SIZE;

            // h_stream_output[stream_id]에 결과가 이미 들어와 있음
            for (int b = 0; b < current_batch_size; b++) {
                float* logits = h_stream_output[stream_id] + (b * NUM_CLASSES);

                // Softmax on CPU
                float max_val = logits[0];
                for (int k = 1; k < NUM_CLASSES; k++) if (logits[k] > max_val) max_val = logits[k];

                float sum = 0.0f;
                for (int k = 0; k < NUM_CLASSES; k++) {
                    probabilities[img_start_idx + b][k] = exp(logits[k] - max_val);
                    sum += probabilities[img_start_idx + b][k];
                }
                for (int k = 0; k < NUM_CLASSES; k++) probabilities[img_start_idx + b][k] /= sum;
            }
        }

        // =========================================================
        // Step 2: 새로운 작업 제출 (GPU Enqueue)
        // =========================================================
        int new_batch_idx = step;

        if (new_batch_idx < total_iterations) {
            int img_start_idx = new_batch_idx * BATCH_SIZE;
            int current_batch_size = (img_start_idx + BATCH_SIZE > image->n) ? (image->n - img_start_idx) : BATCH_SIZE;
            size_t one_img_bytes = sizeof(float) * 3 * IMG_SIZE * IMG_SIZE;

            // (A) Data Packing (Host -> Host Temp)
            // 비동기 전송 중 원본 image 포인터가 불안정할 수 있으므로, 
            // 안전한 스트림 전용 버퍼(h_stream_input)에 복사해둡니다.
            for (int b = 0; b < current_batch_size; b++) {
                memcpy(h_stream_input[stream_id] + (b * 3 * IMG_SIZE * IMG_SIZE),
                    image[img_start_idx + b].data, one_img_bytes);
            }

            // (B) Async Write (Host -> Device) : CL_FALSE !!
            // CPU는 여기서 멈추지 않고 바로 넘어갑니다.
            clEnqueueWriteBuffer(q, d_input_img[stream_id], CL_FALSE, 0,
                one_img_bytes * current_batch_size, h_stream_input[stream_id], 0, NULL, NULL);


            // (C) Kernel Execution (Batch Processing)
            // [중요] 모든 clSetKernelArg에 스트림 전용 버퍼(d_buf...[stream_id])를 써야 함

            // --- Patch Embed ---
            clSetKernelArg(k_conv2d, 0, sizeof(cl_mem), &d_input_img[stream_id]);
            clSetKernelArg(k_conv2d, 1, sizeof(cl_mem), &d_buf2[stream_id]);
            clSetKernelArg(k_conv2d, 2, sizeof(cl_mem), &d_networks[1]); // Weight (공유)
            clSetKernelArg(k_conv2d, 3, sizeof(cl_mem), &d_networks[2]); // Bias (공유)
            size_t gws_conv[1] = { (size_t)current_batch_size * num_patches * dim };
            clEnqueueNDRangeKernel(q, k_conv2d, 1, NULL, gws_conv, NULL, 0, NULL, NULL);

            // --- Prep ---
            clSetKernelArg(k_prep, 0, sizeof(cl_mem), &d_buf2[stream_id]);
            clSetKernelArg(k_prep, 1, sizeof(cl_mem), &d_buf1[stream_id]);
            clSetKernelArg(k_prep, 2, sizeof(cl_mem), &d_networks[0]);
            clSetKernelArg(k_prep, 3, sizeof(cl_mem), &d_networks[3]);
            size_t gws_prep[1] = { (size_t)current_batch_size * tokens_per_img * dim };
            clEnqueueNDRangeKernel(q, k_prep, 1, NULL, gws_prep, NULL, 0, NULL, NULL);

            // --- Encoder Loop ---
            int net_idx = 4;
            int tokens_in_pass = tokens_per_img * current_batch_size;

            for (int l = 0; l < 12; l++) {
                // LN1
                clSetKernelArg(k_ln, 0, sizeof(cl_mem), &d_buf1[stream_id]);
                clSetKernelArg(k_ln, 1, sizeof(cl_mem), &d_buf2[stream_id]);
                clSetKernelArg(k_ln, 2, sizeof(cl_mem), &d_networks[net_idx + 0]);
                clSetKernelArg(k_ln, 3, sizeof(cl_mem), &d_networks[net_idx + 1]);
                size_t gws_ln[1] = { (size_t)tokens_in_pass };
                clEnqueueNDRangeKernel(q, k_ln, 1, NULL, gws_ln, NULL, 0, NULL, NULL);

                // MHA - Linear1
                set_linear_args(k_linear, d_buf2[stream_id], d_intermediate[stream_id],
                    d_networks[net_idx + 2], d_networks[net_idx + 3], dim, dim * 3, tokens_in_pass);
                global_linear[0] = ((tokens_in_pass + 15) / 16) * 16;
                global_linear[1] = ((dim * 3 + 15) / 16) * 16;
                clEnqueueNDRangeKernel(q, k_linear, 2, NULL, global_linear, local_linear, 0, NULL, NULL);

                // MHA - Score
                clSetKernelArg(k_attn_score, 0, sizeof(cl_mem), &d_intermediate[stream_id]);
                clSetKernelArg(k_attn_score, 1, sizeof(cl_mem), &d_scores[stream_id]);
                clSetKernelArg(k_attn_score, 2, sizeof(int), &tokens_per_img);
                size_t gws_score[3] = { (size_t)current_batch_size * NUM_HEADS, (size_t)tokens_per_img, (size_t)tokens_per_img };
                clEnqueueNDRangeKernel(q, k_attn_score, 3, NULL, gws_score, NULL, 0, NULL, NULL);

                // MHA - Softmax
                clSetKernelArg(k_softmax, 0, sizeof(cl_mem), &d_scores[stream_id]);
                clSetKernelArg(k_softmax, 1, sizeof(int), &tokens_per_img);
                size_t gws_softmax[2] = { (size_t)current_batch_size * NUM_HEADS, (size_t)tokens_per_img };
                clEnqueueNDRangeKernel(q, k_softmax, 2, NULL, gws_softmax, NULL, 0, NULL, NULL);

                // MHA - Value
                clSetKernelArg(k_attn_val, 0, sizeof(cl_mem), &d_scores[stream_id]);
                clSetKernelArg(k_attn_val, 1, sizeof(cl_mem), &d_intermediate[stream_id]);
                clSetKernelArg(k_attn_val, 2, sizeof(cl_mem), &d_buf2[stream_id]);
                clSetKernelArg(k_attn_val, 3, sizeof(int), &tokens_per_img);
                size_t gws_val[3] = { (size_t)current_batch_size * NUM_HEADS, (size_t)tokens_per_img, (size_t)HEAD_DIM / 4 };
                clEnqueueNDRangeKernel(q, k_attn_val, 3, NULL, gws_val, NULL, 0, NULL, NULL);

                // MHA - Final Linear
                set_linear_args(k_linear, d_buf2[stream_id], d_residual[stream_id],
                    d_networks[net_idx + 4], d_networks[net_idx + 5], dim, dim, tokens_in_pass);
                global_linear[0] = ((tokens_in_pass + 15) / 16) * 16;
                global_linear[1] = ((dim + 15) / 16) * 16;
                clEnqueueNDRangeKernel(q, k_linear, 2, NULL, global_linear, local_linear, 0, NULL, NULL);

                // Add1
                clSetKernelArg(k_add, 0, sizeof(cl_mem), &d_residual[stream_id]);
                clSetKernelArg(k_add, 1, sizeof(cl_mem), &d_buf1[stream_id]);
                size_t gws_add[1] = { (size_t)tokens_in_pass * dim };
                clEnqueueNDRangeKernel(q, k_add, 1, NULL, gws_add, NULL, 0, NULL, NULL);

                // MLP Part ... (생략 없이 동일하게 진행)
                // LN2
                clSetKernelArg(k_ln, 0, sizeof(cl_mem), &d_buf1[stream_id]);
                clSetKernelArg(k_ln, 1, sizeof(cl_mem), &d_buf2[stream_id]);
                clSetKernelArg(k_ln, 2, sizeof(cl_mem), &d_networks[net_idx + 6]);
                clSetKernelArg(k_ln, 3, sizeof(cl_mem), &d_networks[net_idx + 7]);
                clEnqueueNDRangeKernel(q, k_ln, 1, NULL, gws_ln, NULL, 0, NULL, NULL);

                // 1. FC1 + GELU Fused
                // Input: d_buf2, Output: d_intermediate
                set_linear_args(k_linear_gelu, d_buf2[stream_id], d_intermediate[stream_id],
                    d_networks[net_idx + 8], d_networks[net_idx + 9],
                    dim, hidden_dim, tokens_in_pass);

                global_linear[0] = ((tokens_in_pass + 15) / 16) * 16;
                global_linear[1] = ((hidden_dim + 15) / 16) * 16;

                // k_linear 대신 k_linear_gelu 실행 (GELU 커널 실행은 삭제)
                clEnqueueNDRangeKernel(q, k_linear_gelu, 2, NULL, global_linear, local_linear, 0, NULL, NULL);

                // 2. FC2 + Add Fused
                // Input: d_intermediate
                // Output: d_buf1 (여기에 결과가 누적됨)
                // Prev: d_buf1 (기존 값, Residual Connection)

                // 인자 설정 (k_linear_add는 인자가 하나 더 많음)
                clSetKernelArg(k_linear_add, 0, sizeof(cl_mem), &d_intermediate[stream_id]); // Input
                clSetKernelArg(k_linear_add, 1, sizeof(cl_mem), &d_buf1[stream_id]);       // Output
                clSetKernelArg(k_linear_add, 2, sizeof(cl_mem), &d_networks[net_idx + 10]); // Weight
                clSetKernelArg(k_linear_add, 3, sizeof(cl_mem), &d_networks[net_idx + 11]); // Bias
                clSetKernelArg(k_linear_add, 4, sizeof(int), &hidden_dim); // In features
                clSetKernelArg(k_linear_add, 5, sizeof(int), &dim);        // Out features
                clSetKernelArg(k_linear_add, 6, sizeof(int), &tokens_in_pass);
                clSetKernelArg(k_linear_add, 7, sizeof(cl_mem), &d_buf1[stream_id]);       // Prev State

                global_linear[0] = ((tokens_in_pass + 15) / 16) * 16;
                global_linear[1] = ((dim + 15) / 16) * 16;

                clEnqueueNDRangeKernel(q, k_linear_add, 2, NULL, global_linear, local_linear, 0, NULL, NULL);

                net_idx += 12;
            } // end encoder loop

            // --- Final LN ---
            clSetKernelArg(k_ln, 0, sizeof(cl_mem), &d_buf1[stream_id]);
            clSetKernelArg(k_ln, 1, sizeof(cl_mem), &d_buf2[stream_id]);
            clSetKernelArg(k_ln, 2, sizeof(cl_mem), &d_networks[148]);
            clSetKernelArg(k_ln, 3, sizeof(cl_mem), &d_networks[149]);
            size_t gws_ln_final[1] = { (size_t)tokens_in_pass };
            clEnqueueNDRangeKernel(q, k_ln, 1, NULL, gws_ln_final, NULL, 0, NULL, NULL);

            // --- Gather & Head ---
            clSetKernelArg(k_gather, 0, sizeof(cl_mem), &d_buf2[stream_id]);
            clSetKernelArg(k_gather, 1, sizeof(cl_mem), &d_cls_gathered[stream_id]);
            clSetKernelArg(k_gather, 2, sizeof(int), &tokens_per_img);
            size_t gws_gather[1] = { (size_t)current_batch_size * dim };
            clEnqueueNDRangeKernel(q, k_gather, 1, NULL, gws_gather, NULL, 0, NULL, NULL);

            set_linear_args(k_linear, d_cls_gathered[stream_id], d_cls_out[stream_id],
                d_networks[150], d_networks[151], dim, NUM_CLASSES, current_batch_size);
            global_linear[0] = ((current_batch_size + 15) / 16) * 16;
            global_linear[1] = ((NUM_CLASSES + 15) / 16) * 16;
            clEnqueueNDRangeKernel(q, k_linear, 2, NULL, global_linear, local_linear, 0, NULL, NULL);


            // (D) Async Read (Device -> Host) : CL_FALSE !!
            clEnqueueReadBuffer(q, d_cls_out[stream_id], CL_FALSE, 0,
                sizeof(float) * current_batch_size * NUM_CLASSES, h_stream_output[stream_id], 0, NULL, NULL);
        }
    }

    // ------------------------------------------------------------
    // 5. Cleanup (모든 자원 해제)
    // ------------------------------------------------------------
    // 5. 리소스 해제 (반복문 사용)
    for (int i = 0; i < NUM_STREAMS; i++) {
        clReleaseCommandQueue(queues[i]);
        clReleaseMemObject(d_input_img[i]);
        clReleaseMemObject(d_buf1[i]);
        clReleaseMemObject(d_buf2[i]);
        clReleaseMemObject(d_residual[i]);
        clReleaseMemObject(d_intermediate[i]);
        clReleaseMemObject(d_scores[i]);
        clReleaseMemObject(d_cls_out[i]);
    }

    clReleaseKernel(k_conv2d);
    clReleaseKernel(k_prep);
    clReleaseKernel(k_ln);
    clReleaseKernel(k_linear);
    clReleaseKernel(k_add);
    clReleaseKernel(k_gelu);
    clReleaseKernel(k_attn_score);
    clReleaseKernel(k_softmax);
    clReleaseKernel(k_attn_val);
    clReleaseKernel(k_linear_gelu);
    clReleaseKernel(k_linear_add);
}