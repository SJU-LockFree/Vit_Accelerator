#include <stdio.h>
#include <stdlib.h>
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

// linear_kernel에서 쓰는 타일 크기와 맞춰야 함 (kernel.cl 의 TS와 동일)
#define TS 16

static double now_ms(void) {
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
}

// Helper to set standard linear args
void set_linear_args(cl_kernel kernel, cl_mem in, cl_mem out, cl_mem w, cl_mem b,
    int in_f, int out_f, int num_tokens) {
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

    // 0. 기존 queue는 compute용, transfer용 queue 새로 생성
    cl_device_id device;
    err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,
        sizeof(device), &device, NULL);
    if (err != CL_SUCCESS) {
        printf("clGetCommandQueueInfo failed: %d\n", err);
        return;
    }

    // 전송 큐는 프로파일링 켜서 업로드 시간 측정에 사용
    cl_command_queue queue_xfer =
        clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        printf("clCreateCommandQueue (xfer) failed: %d\n", err);
        return;
    }

    int n_images = image->n;
    if (n_images <= 0) {
        clReleaseCommandQueue(queue_xfer);
        return;
    }

    size_t image_bytes = sizeof(float) * 3 * IMG_SIZE * IMG_SIZE;

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

    // 입력 이미지 버퍼 (더블 버퍼링)
    cl_mem d_input_img[2];
    d_input_img[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, image_bytes, NULL, &err);
    d_input_img[1] = clCreateBuffer(context, CL_MEM_READ_ONLY, image_bytes, NULL, &err);

    // 메인 버퍼 (Ping-Pong용)
    cl_mem d_buf1 = clCreateBuffer(context, CL_MEM_READ_WRITE, size_token_hidden, NULL, &err);
    cl_mem d_buf2 = clCreateBuffer(context, CL_MEM_READ_WRITE, size_token_hidden, NULL, &err);
    cl_mem d_residual = clCreateBuffer(context, CL_MEM_READ_WRITE, size_token_dim, NULL, &err);

    // QKV + MLP Hidden용
    cl_mem d_intermediate = clCreateBuffer(context, CL_MEM_READ_WRITE, size_token_hidden, NULL, &err);

    // Attention Score 버퍼
    cl_mem d_scores = clCreateBuffer(context, CL_MEM_READ_WRITE, size_scores, NULL, &err);

    // 최종 Output 버퍼
    cl_mem d_cls_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * NUM_CLASSES, NULL, &err);

    // 업로드 이벤트 배열 (이미지별)
    cl_event* ev_upload = (cl_event*)calloc(n_images, sizeof(cl_event));

    // ---- 첫 번째 이미지 미리 업로드 (비동기) ----
    err = clEnqueueWriteBuffer(queue_xfer,
        d_input_img[0],          // buf index 0
        CL_FALSE,                // non-blocking
        0,
        image_bytes,
        &image[0].data[0],
        0, NULL,
        &ev_upload[0]);
    if (err != CL_SUCCESS) {
        printf("Initial clEnqueueWriteBuffer failed: %d\n", err);
    }

    // 3. Inference Loop
    for (int img_idx = 0; img_idx < n_images; img_idx++) {

        int buf_idx = img_idx % 2; // 현재 이미지가 사용할 버퍼 인덱스

        // 타이머 변수들 (이미지 하나 기준)
        double t_upload = 0.0;
        double t_patch_embed = 0.0;
        double t_encoder = 0.0;
        double t_final_ln = 0.0;
        double t_head = 0.0;
        double t_read = 0.0;
        double t_softmax = 0.0;
        double t0, t1;

        // -----------------------------
        // (B) Patch Embedding (Conv2d + Prep 전체 시간)
        //   - conv2d는 현재 이미지 업로드 완료 이벤트(ev_upload[img_idx])를 기다린 뒤 실행
        // -----------------------------
        t0 = now_ms();

        // conv2d: input = d_input_img[buf_idx]
        clSetKernelArg(k_conv2d, 0, sizeof(cl_mem), &d_input_img[buf_idx]);
        clSetKernelArg(k_conv2d, 1, sizeof(cl_mem), &d_buf2);
        clSetKernelArg(k_conv2d, 2, sizeof(cl_mem), &d_networks[1]);
        clSetKernelArg(k_conv2d, 3, sizeof(cl_mem), &d_networks[2]);
        size_t gws_conv[1] = { (size_t)(num_patches * dim) };

        cl_event ev_conv = NULL;
        if (ev_upload[img_idx] != NULL) {
            // 업로드 완료 후에 conv2d 실행
            clEnqueueNDRangeKernel(queue, k_conv2d, 1, NULL, gws_conv, NULL,
                1, &ev_upload[img_idx], &ev_conv);
        }
        else {
            // (이론상 없어야 하지만 방어 코드)
            clEnqueueNDRangeKernel(queue, k_conv2d, 1, NULL, gws_conv, NULL,
                0, NULL, &ev_conv);
        }

        // prepare_input_kernel: conv2d 이후 실행
        clSetKernelArg(k_prep, 0, sizeof(cl_mem), &d_buf2);
        clSetKernelArg(k_prep, 1, sizeof(cl_mem), &d_buf1);
        clSetKernelArg(k_prep, 2, sizeof(cl_mem), &d_networks[0]);
        clSetKernelArg(k_prep, 3, sizeof(cl_mem), &d_networks[3]);
        size_t gws_prep[1] = { (size_t)(tokens * dim) };
        cl_event ev_prep = NULL;
        clEnqueueNDRangeKernel(queue, k_prep, 1, NULL, gws_prep, NULL,
            1, &ev_conv, &ev_prep);

        // Patch Embedding 관련 커널이 끝날 때까지 대기 후 시간 측정
        clFinish(queue);
        t1 = now_ms();
        t_patch_embed += (t1 - t0);

        // 업로드 시간 측정: ev_upload[img_idx]의 프로파일링 정보 사용
        if (ev_upload[img_idx] != NULL) {
            cl_ulong start = 0, end = 0;
            clGetEventProfilingInfo(ev_upload[img_idx],
                CL_PROFILING_COMMAND_START,
                sizeof(start), &start, NULL);
            clGetEventProfilingInfo(ev_upload[img_idx],
                CL_PROFILING_COMMAND_END,
                sizeof(end), &end, NULL);
            t_upload = (double)(end - start) * 1e-6; // ns -> ms
            clReleaseEvent(ev_upload[img_idx]);
            ev_upload[img_idx] = NULL;
        }

        // conv / prep 이벤트도 해제
        if (ev_conv)  clReleaseEvent(ev_conv);
        if (ev_prep)  clReleaseEvent(ev_prep);

        // -----------------------------
        // (D) Encoder Layers 전체 시간
        // -----------------------------
        t0 = now_ms();

        int net_idx = 4;
        for (int i = 0; i < 12; i++) {
            // --- LN1 ---
            clSetKernelArg(k_ln, 0, sizeof(cl_mem), &d_buf1);
            clSetKernelArg(k_ln, 1, sizeof(cl_mem), &d_buf2);
            clSetKernelArg(k_ln, 2, sizeof(cl_mem), &d_networks[net_idx + 0]);
            clSetKernelArg(k_ln, 3, sizeof(cl_mem), &d_networks[net_idx + 1]);
            size_t gws_ln[1] = { (size_t)tokens };
            clEnqueueNDRangeKernel(queue, k_ln, 1, NULL, gws_ln, NULL, 0, NULL, NULL);

            // --- Multi-Head Attention ---
            // 1. QKV Linear Projection: d_buf2 -> d_intermediate
            {
                int in_f = dim;
                int out_f = dim * 3;
                set_linear_args(k_linear, d_buf2, d_intermediate,
                    d_networks[net_idx + 2], d_networks[net_idx + 3],
                    in_f, out_f, tokens);

                size_t lws_linear[2] = { TS, TS };
                size_t gws_linear_qkv[2] = {
                    (size_t)((tokens + TS - 1) / TS * TS),
                    (size_t)((out_f + TS - 1) / TS * TS)
                };
                clEnqueueNDRangeKernel(queue, k_linear, 2, NULL,
                    gws_linear_qkv, lws_linear,
                    0, NULL, NULL);
            }

            // 2. Scores(QK^T)
            clSetKernelArg(k_attn_score, 0, sizeof(cl_mem), &d_intermediate);
            clSetKernelArg(k_attn_score, 1, sizeof(cl_mem), &d_scores);
            clSetKernelArg(k_attn_score, 2, sizeof(int), &tokens);
            size_t gws_score[3] = { NUM_HEADS, (size_t)tokens, (size_t)tokens };
            clEnqueueNDRangeKernel(queue, k_attn_score, 3, NULL, gws_score, NULL, 0, NULL, NULL);

            // 3. Softmax
            clSetKernelArg(k_softmax, 0, sizeof(cl_mem), &d_scores);
            clSetKernelArg(k_softmax, 1, sizeof(int), &tokens);
            size_t gws_softmax_attn[2] = { NUM_HEADS, (size_t)tokens };
            clEnqueueNDRangeKernel(queue, k_softmax, 2, NULL, gws_softmax_attn, NULL, 0, NULL, NULL);

            // 4. Values (Scores * V) -> d_buf2
            clSetKernelArg(k_attn_val, 0, sizeof(cl_mem), &d_scores);
            clSetKernelArg(k_attn_val, 1, sizeof(cl_mem), &d_intermediate);
            clSetKernelArg(k_attn_val, 2, sizeof(cl_mem), &d_buf2);
            clSetKernelArg(k_attn_val, 3, sizeof(int), &tokens);
            size_t gws_val[3] = { NUM_HEADS, (size_t)tokens, HEAD_DIM };
            clEnqueueNDRangeKernel(queue, k_attn_val, 3, NULL, gws_val, NULL, 0, NULL, NULL);

            // 5. Output Projection -> d_residual
            {
                int in_f = dim;
                int out_f = dim;
                set_linear_args(k_linear, d_buf2, d_residual,
                    d_networks[net_idx + 4], d_networks[net_idx + 5],
                    in_f, out_f, tokens);

                size_t lws_linear[2] = { TS, TS };
                size_t gws_linear_proj[2] = {
                    (size_t)((tokens + TS - 1) / TS * TS),
                    (size_t)((out_f + TS - 1) / TS * TS)
                };
                clEnqueueNDRangeKernel(queue, k_linear, 2, NULL,
                    gws_linear_proj, lws_linear,
                    0, NULL, NULL);
            }

            // --- Residual Add 1 ---
            clSetKernelArg(k_add, 0, sizeof(cl_mem), &d_residual);
            clSetKernelArg(k_add, 1, sizeof(cl_mem), &d_buf1);
            size_t gws_add[1] = { (size_t)(tokens * dim) };
            clEnqueueNDRangeKernel(queue, k_add, 1, NULL, gws_add, NULL, 0, NULL, NULL);

            // --- LN2 ---
            clSetKernelArg(k_ln, 0, sizeof(cl_mem), &d_buf1);
            clSetKernelArg(k_ln, 1, sizeof(cl_mem), &d_buf2);
            clSetKernelArg(k_ln, 2, sizeof(cl_mem), &d_networks[net_idx + 6]);
            clSetKernelArg(k_ln, 3, sizeof(cl_mem), &d_networks[net_idx + 7]);
            clEnqueueNDRangeKernel(queue, k_ln, 1, NULL, gws_ln, NULL, 0, NULL, NULL);

            // --- MLP ---
            // 1. FC1: d_buf2 -> d_intermediate
            {
                int in_f = dim;
                int out_f = hidden_dim;
                set_linear_args(k_linear, d_buf2, d_intermediate,
                    d_networks[net_idx + 8], d_networks[net_idx + 9],
                    in_f, out_f, tokens);

                size_t lws_linear[2] = { TS, TS };
                size_t gws_mlp1[2] = {
                    (size_t)((tokens + TS - 1) / TS * TS),
                    (size_t)((out_f + TS - 1) / TS * TS)
                };
                clEnqueueNDRangeKernel(queue, k_linear, 2, NULL,
                    gws_mlp1, lws_linear,
                    0, NULL, NULL);
            }

            // 2. GELU
            clSetKernelArg(k_gelu, 0, sizeof(cl_mem), &d_intermediate);
            size_t gws_gelu[1] = { (size_t)(tokens * hidden_dim) };
            clEnqueueNDRangeKernel(queue, k_gelu, 1, NULL, gws_gelu, NULL, 0, NULL, NULL);

            // 3. FC2: d_intermediate -> d_residual
            {
                int in_f = hidden_dim;
                int out_f = dim;
                set_linear_args(k_linear, d_intermediate, d_residual,
                    d_networks[net_idx + 10], d_networks[net_idx + 11],
                    in_f, out_f, tokens);

                size_t lws_linear[2] = { TS, TS };
                size_t gws_mlp2[2] = {
                    (size_t)((tokens + TS - 1) / TS * TS),
                    (size_t)((out_f + TS - 1) / TS * TS)
                };
                clEnqueueNDRangeKernel(queue, k_linear, 2, NULL,
                    gws_mlp2, lws_linear,
                    0, NULL, NULL);
            }

            // --- Residual Add 2 ---
            clSetKernelArg(k_add, 0, sizeof(cl_mem), &d_residual);
            clSetKernelArg(k_add, 1, sizeof(cl_mem), &d_buf1);
            clEnqueueNDRangeKernel(queue, k_add, 1, NULL, gws_add, NULL, 0, NULL, NULL);

            net_idx += 12;
        }

        clFinish(queue);
        t1 = now_ms();
        t_encoder += (t1 - t0);

        // (E) Final Layer Norm
        t0 = now_ms();
        clSetKernelArg(k_ln, 0, sizeof(cl_mem), &d_buf1);
        clSetKernelArg(k_ln, 1, sizeof(cl_mem), &d_buf2);
        clSetKernelArg(k_ln, 2, sizeof(cl_mem), &d_networks[148]);
        clSetKernelArg(k_ln, 3, sizeof(cl_mem), &d_networks[149]);
        size_t gws_ln_final[1] = { (size_t)tokens };
        clEnqueueNDRangeKernel(queue, k_ln, 1, NULL, gws_ln_final, NULL, 0, NULL, NULL);
        clFinish(queue);
        t1 = now_ms();
        t_final_ln += (t1 - t0);

        // (F) Classifier Head (CLS 토큰만 사용 → tokens=1 로 설정)
        t0 = now_ms();
        {
            int in_f = dim;
            int out_f = NUM_CLASSES;
            int head_tokens = 1; // CLS 한 개만 사용

            set_linear_args(k_linear, d_buf2, d_cls_out,
                d_networks[150], d_networks[151],
                in_f, out_f, head_tokens);

            size_t lws_linear[2] = { TS, TS };
            size_t gws_head[2] = {
                (size_t)((head_tokens + TS - 1) / TS * TS),   // 최소 16
                (size_t)((out_f + TS - 1) / TS * TS)
            };
            clEnqueueNDRangeKernel(queue, k_linear, 2, NULL,
                gws_head, lws_linear,
                0, NULL, NULL);
        }
        clFinish(queue);
        t1 = now_ms();
        t_head += (t1 - t0);

        // (G) Read Result
        t0 = now_ms();
        clEnqueueReadBuffer(queue, d_cls_out, CL_TRUE, 0,
            sizeof(float) * NUM_CLASSES,
            probabilities[img_idx], 0, NULL, NULL);
        t1 = now_ms();
        t_read += (t1 - t0);

        // Softmax (CPU 계산 - 정확도 유지용)
        t0 = now_ms();
        float max_val = probabilities[img_idx][0];
        for (int k = 1; k < NUM_CLASSES; k++)
            if (probabilities[img_idx][k] > max_val) max_val = probabilities[img_idx][k];

        float sum_exp = 0.0f;
        for (int k = 0; k < NUM_CLASSES; k++) {
            probabilities[img_idx][k] = exp(probabilities[img_idx][k] - max_val);
            sum_exp += probabilities[img_idx][k];
        }
        for (int k = 0; k < NUM_CLASSES; k++)
            probabilities[img_idx][k] /= sum_exp;
        t1 = now_ms();
        t_softmax += (t1 - t0);

        // -----------------------------
        // 다음 이미지 업로드를 미리 걸어둠 (overlap)
        // -----------------------------
        if (img_idx + 1 < n_images) {
            int next = img_idx + 1;
            int next_buf_idx = next % 2;

            err = clEnqueueWriteBuffer(queue_xfer,
                d_input_img[next_buf_idx],
                CL_FALSE,
                0,
                image_bytes,
                &image[next].data[0],
                0, NULL,
                &ev_upload[next]);
            if (err != CL_SUCCESS) {
                printf("clEnqueueWriteBuffer (img %d) failed: %d\n", next, err);
            }
        }

        // 이미지별 타이밍 출력
        double t_total = t_upload + t_patch_embed + t_encoder
            + t_final_ln + t_head + t_read + t_softmax;

        printf("[IMG %d] upload=%.3f ms, patch=%.3f ms, encoder=%.3f ms, "
            "final_ln=%.3f ms, head=%.3f ms, read=%.3f ms, softmax=%.3f ms, total=%.3f ms\n",
            img_idx,
            t_upload, t_patch_embed, t_encoder,
            t_final_ln, t_head, t_read, t_softmax, t_total);
    }

    // 4. Clean up
    free(ev_upload);

    clReleaseMemObject(d_input_img[0]);
    clReleaseMemObject(d_input_img[1]);
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

    clReleaseCommandQueue(queue_xfer);
}
