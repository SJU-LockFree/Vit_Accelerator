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
#define NUM_STREAMS 1
#define BATCH_SIZE 32


double t_upload = 0.0;
double t_patch_embed = 0.0;
double t_encoder = 0.0;
double t_final_ln = 0.0;
double t_head = 0.0;
double t_read = 0.0;
double t_softmax = 0.0;
double t0, t1, ts, te;

static double now_ms(void) {
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
}

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

/* 시간 측정 단위 초기화 */
void reset_timer() {
    t_upload = 0.0;
    t_patch_embed = 0.0;
    t_encoder = 0.0;
    t_final_ln = 0.0;
    t_head = 0.0;
    t_read = 0.0;
    t_softmax = 0.0;
}

/* [ 수정 ] : 디바이스를 추가 인자로 받음 */
void ViT_opencl(ImageData* image, cl_mem* d_networks, float** probabilities,
    cl_context context, cl_command_queue queue, cl_program program, cl_device_id device)
{
    // 0. 기타 변수 생성
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
    cl_kernel k_linear_gelu = clCreateKernel(program, "linear_gelu_kernel", &err);
    cl_kernel k_linear_add = clCreateKernel(program, "linear_add_kernel", &err);
    cl_kernel k_gather = clCreateKernel(program, "gather_cls_kernel", &err);
    

    // 2. 버퍼 크기 계산
    int num_patches = (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE); // 196
    int tokens_per_img = num_patches + 1;                                   // 197
    int dim = EMBED_DIM;                                         // 768
    int hidden_dim = dim * MLP_RATIO;                                   // 3072

    int max_tokens = BATCH_SIZE * tokens_per_img;

    // 버퍼 사이즈 정의
    size_t size_input_batch = sizeof(float) * BATCH_SIZE * 3 * IMG_SIZE * IMG_SIZE;
    size_t size_token_dim = sizeof(float) * max_tokens * dim;
    size_t size_token_hidden = sizeof(float) * max_tokens * hidden_dim;
    size_t size_scores = sizeof(float) * BATCH_SIZE * NUM_HEADS * tokens_per_img * tokens_per_img;
    size_t size_cls_out = sizeof(float) * BATCH_SIZE * NUM_CLASSES;

    size_t local_linear[2] = { 16, 16 };
    size_t global_linear[2];


    /* [ 수정 ] */
    // 3. 스트림별 리소스 할당 (배열로 선언)
    cl_command_queue queues[NUM_STREAMS];
    cl_mem d_input_img[NUM_STREAMS];        // 입력 이미지 버퍼
    cl_mem d_buf1[NUM_STREAMS];             // 메인 버퍼 (Ping-Pong용)
    cl_mem d_buf2[NUM_STREAMS];             
    cl_mem d_residual[NUM_STREAMS];
    cl_mem d_intermediate[NUM_STREAMS];     // ★수정됨★: QKV 뿐만 아니라 MLP의 Hidden Layer(4배)도 담을 수 있도록 가장 큰 크기로 할당
    cl_mem d_scores[NUM_STREAMS];           // Attention Score 버퍼
    cl_mem d_cls_out[NUM_STREAMS];          // 최종 Output 버퍼

    // 커맨드 큐와 버퍼 생성
    for (int i = 0; i < NUM_STREAMS; i++) {
        // 커맨드 큐 생성 (Out-of-order가 아닌 일반 큐도 무방, 여기선 독립된 큐 사용)
        queues[i] = clCreateCommandQueueWithProperties(context, device, 0, &err);

        d_input_img[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, size_input_batch, NULL, &err);
        d_buf1[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, size_token_hidden, NULL, &err);
        d_buf2[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, size_token_hidden, NULL, &err);
        d_residual[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, size_token_dim, NULL, &err);
        d_intermediate[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, size_token_hidden, NULL, &err);
        d_scores[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, size_scores, NULL, &err);
        d_cls_out[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_cls_out, NULL, &err);
    }
    float* h_batch = (float*)malloc(size_input_batch);
    float* h_logits[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        h_logits[i] = (float*)malloc(sizeof(float) * BATCH_SIZE * NUM_CLASSES);
    }
   
    // 4. Inference Loop
    int total_batches = (image->n + BATCH_SIZE - 1) / BATCH_SIZE;

    for (int i = 0; i < total_batches + NUM_STREAMS; ++i) {
        int stream_id = i % NUM_STREAMS;

        // [SYNC] 해당 스트림의 이전 작업이 끝날 때까지 대기
        // (이 시점에서 이전 이미지는 d_cls_out에 결과가 들어와 있고, 전송도 끝난 상태임)
        clFinish(queues[stream_id]);

        // [CPU Post-processing] 완료된 이전 이미지의 Softmax 계산
        // 현재 인덱스 i에서 NUM_STREAMS만큼 뺀 인덱스가 방금 완료된 이미지임
        int finished_batch = i - NUM_STREAMS;
        if (finished_batch >= 0 && finished_batch < total_batches) {
            int img_start = finished_batch * BATCH_SIZE;
            int cur_bs = (img_start + BATCH_SIZE > image->n)
                ? (image->n - img_start)
                : BATCH_SIZE;

            int finished_stream = finished_batch % NUM_STREAMS;
            if (finished_stream < 0) finished_stream += NUM_STREAMS; // 안전용


            // 이 스트림에서 방금 읽어온 배치 logits
            float* logits_batch = h_logits[finished_stream];

            for (int b = 0; b < cur_bs; ++b) {
                float* logits = logits_batch + b * NUM_CLASSES;

                // softmax 계산
                float max_val = logits[0];
                for (int k = 1; k < NUM_CLASSES; ++k)
                    if (logits[k] > max_val) max_val = logits[k];

                float sum = 0.f;
                for (int k = 0; k < NUM_CLASSES; ++k) {
                    float v = expf(logits[k] - max_val);
                    probabilities[img_start + b][k] = v;
                    sum += v;
                }
                float inv_sum = 1.0f / sum;
                for (int k = 0; k < NUM_CLASSES; ++k)
                    probabilities[img_start + b][k] *= inv_sum;
            }
        }


        // 타이머 변수 리셋
        reset_timer();

        // [New Work] 처리할 이미지가 남았다면 새로운 작업 Enqueue
        if (i < total_batches) {
            int img_start = i * BATCH_SIZE;
            int cur_bs = (img_start + BATCH_SIZE > image->n)
                ? (image->n - img_start)
                : BATCH_SIZE;

            // Host에서 h_batch에 이미지 cur_bs 개 붙이기
            for (int b = 0; b < cur_bs; ++b) {
                memcpy(h_batch + b * 3 * IMG_SIZE * IMG_SIZE,
                    image[img_start + b].data,
                    sizeof(float) * 3 * IMG_SIZE * IMG_SIZE);
            }

            t0 = now_ms();
            size_t upload_bytes = sizeof(float) * 3 * IMG_SIZE * IMG_SIZE * cur_bs;
            clEnqueueWriteBuffer(queues[stream_id], d_input_img[stream_id], CL_FALSE, 0,
                upload_bytes, h_batch, 0, NULL, NULL);
            t1 = now_ms();
            t_upload += (t1 - t0);

            // (B) Patch Embedding
            size_t gws_conv[1] = { (size_t)cur_bs * num_patches * dim };
            clSetKernelArg(k_conv2d, 0, sizeof(cl_mem), &d_input_img[stream_id]);
            clSetKernelArg(k_conv2d, 1, sizeof(cl_mem), &d_buf2[stream_id]);
            clSetKernelArg(k_conv2d, 2, sizeof(cl_mem), &d_networks[1]);
            clSetKernelArg(k_conv2d, 3, sizeof(cl_mem), &d_networks[2]);
            clSetKernelArg(k_conv2d, 4, sizeof(int), &cur_bs); // ★ batch_size
            clEnqueueNDRangeKernel(queues[stream_id], k_conv2d, 1, NULL, gws_conv, NULL, 0, NULL, NULL);

            // (C) CLS Token + Pos Embedding
            size_t gws_prep[1] = { (size_t)cur_bs * tokens_per_img * dim };

            clSetKernelArg(k_prep, 0, sizeof(cl_mem), &d_buf2[stream_id]);
            clSetKernelArg(k_prep, 1, sizeof(cl_mem), &d_buf1[stream_id]);
            clSetKernelArg(k_prep, 2, sizeof(cl_mem), &d_networks[0]);
            clSetKernelArg(k_prep, 3, sizeof(cl_mem), &d_networks[3]);
            clSetKernelArg(k_prep, 4, sizeof(int), &tokens_per_img);
            clSetKernelArg(k_prep, 5, sizeof(int), &cur_bs);
            clEnqueueNDRangeKernel(queues[stream_id], k_prep, 1, NULL, gws_prep, NULL, 0, NULL, NULL);

            //clFinish(queue);
            t1 = now_ms();
            t_patch_embed += (t1 - t0);

            // (D) Encoder Layers
            t0 = now_ms();

            int net_idx = 4;
            for (int i = 0; i < 12; i++) {
                // --- Multi-Head Attention ---

                /* LN 1 */
                clSetKernelArg(k_ln, 0, sizeof(cl_mem), &d_buf1[stream_id]);
                clSetKernelArg(k_ln, 1, sizeof(cl_mem), &d_buf2[stream_id]);
                clSetKernelArg(k_ln, 2, sizeof(cl_mem), &d_networks[net_idx + 0]);
                clSetKernelArg(k_ln, 3, sizeof(cl_mem), &d_networks[net_idx + 1]);

                // [Change] Work-Group Parallelism
                // Global: Tokens * 256, Local: 256
                int tokens_in_pass = tokens_per_img * cur_bs;

                size_t gws_ln[1] = { (size_t)tokens_in_pass * 256 };
                size_t lws_ln[1] = { 256 };
                clEnqueueNDRangeKernel(queues[stream_id], k_ln, 1, NULL, gws_ln, lws_ln, 0, NULL, NULL);

                /* MHA - Optimized */

                // 1. Linear 1 (QKV Projection)
                // 업그레이드된 linear_kernel 사용 (Padding+Unroll 적용됨)                
                set_linear_args(k_linear, d_buf2[stream_id], d_intermediate[stream_id],
                    d_networks[net_idx + 2], d_networks[net_idx + 3],
                    dim, dim * 3, tokens_in_pass);

                global_linear[0] = ((tokens_in_pass + 15) / 16) * 16;
                global_linear[1] = ((dim * 3 + 15) / 16) * 16;
                clEnqueueNDRangeKernel(queues[stream_id], k_linear, 2, NULL, global_linear, local_linear, 0, NULL, NULL);

                // 2. Score Calculation (Optimized Tiled)      

                clSetKernelArg(k_attn_score, 0, sizeof(cl_mem), &d_intermediate[stream_id]);
                clSetKernelArg(k_attn_score, 1, sizeof(cl_mem), &d_scores[stream_id]);
                clSetKernelArg(k_attn_score, 2, sizeof(int), &tokens_per_img);
                clSetKernelArg(k_attn_score, 3, sizeof(int), &cur_bs);

                size_t global_score[3] = {
                    ((tokens_per_img + 15) / 16) * 16,  // j
                    ((tokens_per_img + 15) / 16) * 16,  // i
                    (size_t)(cur_bs * NUM_HEADS)       // b*h
                };
                size_t local_score[3] = { 16, 16, 1 };

                clEnqueueNDRangeKernel(queues[stream_id], k_attn_score, 3, NULL, global_score, local_score, 0, NULL, NULL);

                // 3. Softmax
                int rows = cur_bs * NUM_HEADS * tokens_per_img;
                size_t gws_softmax[1] = { (size_t)rows * 256 };
                size_t lws_softmax[1] = { 256 };

                clSetKernelArg(k_softmax, 0, sizeof(cl_mem), &d_scores[stream_id]);
                clSetKernelArg(k_softmax, 1, sizeof(int), &tokens_per_img);

                clEnqueueNDRangeKernel(queues[stream_id], k_softmax, 1, NULL, gws_softmax, lws_softmax, 0, NULL, NULL);

                // 4. Values Calculation (Optimized Tiled)
                clSetKernelArg(k_attn_val, 0, sizeof(cl_mem), &d_scores[stream_id]);
                clSetKernelArg(k_attn_val, 1, sizeof(cl_mem), &d_intermediate[stream_id]);
                clSetKernelArg(k_attn_val, 2, sizeof(cl_mem), &d_buf2[stream_id]);
                clSetKernelArg(k_attn_val, 3, sizeof(int), &tokens_per_img);
                clSetKernelArg(k_attn_val, 4, sizeof(int), &cur_bs);

                size_t global_val[3] = {
                    ((HEAD_DIM + 15) / 16) * 16,
                    ((tokens_per_img + 15) / 16) * 16,
                    (size_t)(cur_bs * NUM_HEADS)
                };
                size_t local_val[3] = { 16, 16, 1 };

                clEnqueueNDRangeKernel(queues[stream_id], k_attn_val, 3, NULL, global_val, local_val, 0, NULL, NULL);

                // 5. Final Linear + Add Fused (커널 융합)
                // linear_add_kernel을 재활용합니다.
                // Input: d_buf2 (Attention Output)
                // Output: d_buf1 (Accumulator) -> 결과가 여기에 바로 더해짐
                // Prev_State: d_buf1 (기존 값, Skip Connection)

                clSetKernelArg(k_linear_add, 0, sizeof(cl_mem), &d_buf2[stream_id]);        // Input
                clSetKernelArg(k_linear_add, 1, sizeof(cl_mem), &d_residual[stream_id]);     // Output (d_residual에 임시 저장 후 add가 아님! -> 바로 d_buf1에 더하고 싶지만..)
                

                clSetKernelArg(k_linear_add, 1, sizeof(cl_mem), &d_buf1[stream_id]);         // Output (여기에 결과 저장)
                clSetKernelArg(k_linear_add, 2, sizeof(cl_mem), &d_networks[net_idx + 4]);   // Weight
                clSetKernelArg(k_linear_add, 3, sizeof(cl_mem), &d_networks[net_idx + 5]);   // Bias
                clSetKernelArg(k_linear_add, 4, sizeof(int), &dim);
                clSetKernelArg(k_linear_add, 5, sizeof(int), &dim);
                clSetKernelArg(k_linear_add, 6, sizeof(int), &tokens_in_pass);
                clSetKernelArg(k_linear_add, 7, sizeof(cl_mem), &d_buf1[stream_id]);         // Prev State (더해질 대상)

                global_linear[0] = ((tokens_in_pass + 15) / 16) * 16;
                global_linear[1] = ((dim + 15) / 16) * 16;
                clEnqueueNDRangeKernel(queues[stream_id], k_linear_add, 2, NULL, global_linear, local_linear, 0, NULL, NULL);

                /* LN 2 */ 
                clSetKernelArg(k_ln, 0, sizeof(cl_mem), &d_buf1[stream_id]);
                clSetKernelArg(k_ln, 1, sizeof(cl_mem), &d_buf2[stream_id]);
                clSetKernelArg(k_ln, 2, sizeof(cl_mem), &d_networks[net_idx + 6]);
                clSetKernelArg(k_ln, 3, sizeof(cl_mem), &d_networks[net_idx + 7]);
                // 위에서 정의한 gws_ln, lws_ln 재사용
                clEnqueueNDRangeKernel(queues[stream_id], k_ln, 1, NULL, gws_ln, lws_ln, 0, NULL, NULL);



                /* MLP - Optimized Tiled Kernels */

                // 1. FC1 + GELU
                set_linear_args(k_linear_gelu, d_buf2[stream_id], d_intermediate[stream_id],
                    d_networks[net_idx + 8], d_networks[net_idx + 9],
                    dim, hidden_dim, tokens_in_pass);
                global_linear[0] = ((tokens_in_pass + 15) / 16) * 16;
                global_linear[1] = ((hidden_dim + 15) / 16) * 16;
                clEnqueueNDRangeKernel(queues[stream_id], k_linear_gelu, 2, NULL, global_linear, local_linear, 0, NULL, NULL);


                // 2. FC2 + Add
                clSetKernelArg(k_linear_add, 0, sizeof(cl_mem), &d_intermediate[stream_id]);
                clSetKernelArg(k_linear_add, 1, sizeof(cl_mem), &d_buf1[stream_id]);
                clSetKernelArg(k_linear_add, 2, sizeof(cl_mem), &d_networks[net_idx + 10]);
                clSetKernelArg(k_linear_add, 3, sizeof(cl_mem), &d_networks[net_idx + 11]);
                clSetKernelArg(k_linear_add, 4, sizeof(int), &hidden_dim);
                clSetKernelArg(k_linear_add, 5, sizeof(int), &dim);
                clSetKernelArg(k_linear_add, 6, sizeof(int), &tokens_in_pass);
                clSetKernelArg(k_linear_add, 7, sizeof(cl_mem), &d_buf1[stream_id]);

                global_linear[0] = ((tokens_in_pass + 15) / 16) * 16;
                global_linear[1] = ((dim + 15) / 16) * 16;

                clEnqueueNDRangeKernel(queues[stream_id], k_linear_add, 2,
                    NULL, global_linear, local_linear,
                    0, NULL, NULL);

                net_idx += 12;
            }

            t1 = now_ms();
            t_encoder += (t1 - t0);


            // (E) Final Layer Norm
            t0 = now_ms();

            int tokens_in_pass = tokens_per_img * cur_bs;
            size_t gws_ln_final[1] = { (size_t)tokens_in_pass * 256 };
            size_t lws_ln_final[1] = { 256 };

            clSetKernelArg(k_ln, 0, sizeof(cl_mem), &d_buf1[stream_id]);   // input: final tokens
            clSetKernelArg(k_ln, 1, sizeof(cl_mem), &d_buf2[stream_id]);   // output
            clSetKernelArg(k_ln, 2, sizeof(cl_mem), &d_networks[148]);
            clSetKernelArg(k_ln, 3, sizeof(cl_mem), &d_networks[149]);

            clEnqueueNDRangeKernel(queues[stream_id], k_ln, 1, NULL,
                gws_ln_final, lws_ln_final, 0, NULL, NULL);

            t1 = now_ms();
            t_encoder += (t1 - t0);

            cl_mem d_cls = clCreateBuffer(context, CL_MEM_READ_WRITE,
                sizeof(float) * cur_bs * dim, NULL, &err);

            size_t gws_gather[2] = { (size_t)cur_bs, (size_t)dim };
            size_t lws_gather[2] = { 1, 64 }; 

            clSetKernelArg(k_gather, 0, sizeof(cl_mem), &d_buf2[stream_id]);
            clSetKernelArg(k_gather, 1, sizeof(cl_mem), &d_cls);
            clSetKernelArg(k_gather, 2, sizeof(int), &tokens_per_img);
            clSetKernelArg(k_gather, 3, sizeof(int), &cur_bs);

            clEnqueueNDRangeKernel(queues[stream_id], k_gather, 2, NULL, gws_gather, lws_gather, 0, NULL, NULL);            

            t1 = now_ms();
            t_encoder += (t1 - t0);


            // (F) Classifier Head
            t0 = now_ms();

            set_linear_args(k_linear, d_cls, d_cls_out[stream_id],
                d_networks[150], d_networks[151],
                dim, NUM_CLASSES, cur_bs); 
            global_linear[0] = ((cur_bs + 15) / 16) * 16;
            global_linear[1] = ((NUM_CLASSES + 15) / 16) * 16;
            clEnqueueNDRangeKernel(queues[stream_id], k_linear, 2, NULL, global_linear, local_linear, 0, NULL, NULL);

            t1 = now_ms();
            t_encoder += (t1 - t0);


            // (G) Read Result
            size_t read_bytes = sizeof(float) * cur_bs * NUM_CLASSES;

            t0 = now_ms();
            clEnqueueReadBuffer(queues[stream_id], d_cls_out[stream_id], CL_FALSE, 0,
                read_bytes,
                h_logits[stream_id],   // ← 임시 배치 버퍼
                0, NULL, NULL);
            t1 = now_ms();
            t_read += (t1 - t0);

            clReleaseMemObject(d_cls);
            double t_total = t_upload + t_patch_embed + t_encoder
                + t_final_ln + t_head + t_read + t_softmax;
            /*
            printf("[IMG %d] upload=%.3f ms, patch=%.3f ms, encoder=%.3f ms, "
                "final_ln=%.3f ms, head=%.3f ms, read=%.3f ms, softmax=%.3f ms, total=%.3f ms\n",
                img_idx,
                t_upload, t_patch_embed, t_encoder,
                t_final_ln, t_head, t_read, t_softmax, t_total);
            */
        }
    }

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
        free(h_logits[i]);
    }
    free(h_batch);
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
    clReleaseKernel(k_gather);
}