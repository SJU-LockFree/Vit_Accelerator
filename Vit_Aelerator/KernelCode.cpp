
#define CL_TARGET_OPENCL_VERSION 100
#include "gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void compare(float* ans_seq, float* ans_opencl, int ROW_ans, int COL_ans) {
    float max_diff = 0.0f;
    int mismatch_index = -1;

    const float rtol = 1e-4f;
    const float atol = 1e-5f;

    for (int i = 0; i < ROW_ans * COL_ans; i++) {
        float a = ans_seq[i];
        float b = ans_opencl[i];
        float diff = fabsf(a - b);
        float tol = fmaxf(rtol * fmaxf(fabsf(a), fabsf(b)), atol);

        if (diff > tol) {
            printf("- Sequential version != OpenCL version\n");
            printf("- index %d\tseq=%.9f\tocl=%.9f\tdiff=%.9f\n", i, a, b, diff);
            mismatch_index = i;
            break;
        }

        if (diff > max_diff)
            max_diff = diff;
    }

    if (mismatch_index == -1) {
        printf("\n- Sequential version == OpenCL version\n");
        printf("  (max diff = %.9f)\n", max_diff);
    }
}


int main() {
    float* A, * B, * C_seq, * C_opencl;
    int equal = 1;
    const int ROW_A = 1024, COL_A = 1024, ROW_B = COL_A, COL_B = 1024;

    A = (float*)malloc(sizeof(float) * ROW_A * COL_A);
    B = (float*)malloc(sizeof(float) * ROW_B * COL_B);
    C_seq = (float*)malloc(sizeof(float) * ROW_A * COL_B);
    C_opencl = (float*)malloc(sizeof(float) * ROW_A * COL_B);

    srand(time(NULL));
    for (int i = 0; i < ROW_A * COL_A; i++) A[i] = (float)(rand() % 100) / 100;
    for (int i = 0; i < ROW_B * COL_B; i++) B[i] = (float)(rand() % 100) / 100;

    /*순차적 행렬 연산*/
    printf("Sequential version...\n");
    gemm_seq(A, B, C_seq, ROW_A, COL_A, ROW_B, COL_B);

    printf("\nOpenCL version...\n");
    gemm_opencl(A, B, C_opencl, ROW_A, COL_A, ROW_B, COL_B);

    /*비교*/
    compare(C_seq, C_opencl, ROW_A, COL_B);

    free(A);
    free(B);
    free(C_seq);
    free(C_opencl);

    return 0;
}