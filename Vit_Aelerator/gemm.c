#define _CRT_SECURE_NO_WARNINGS
#include "./gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

char* get_source_code(const char* file_name, size_t* len) {
    FILE* file = fopen(file_name, "rb");
    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    size_t length = (size_t)ftell(file);
    rewind(file);

    char* source_code = (char*)malloc(length + 1);
    fread(source_code, length, 1, file);
    source_code[length] = '\0';
    fclose(file);
    *len = length;

    return source_code;
}

void build_error(cl_program program, cl_device_id device, cl_int err) {
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char* log;

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        CHECK_ERROR(err);

        log = (char*)malloc(log_size + 1);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        CHECK_ERROR(err);

        log[log_size] = '\0';
        printf("Compiler error:\n%s\n", log);
        free(log);
        exit(0);
    };
}

void gemm_seq(const float* A, const float* B, float* C, const int ROW_A, const int COL_A, const int ROW_B, const int COL_B) {

    clock_t start = clock();

    /*
     * 순차 처리 코드 작성
     */

    for (int i = 0; i < ROW_A; i++) {
        for (int j = 0; j < COL_B; j++) {
            C[i * COL_B + j] = 0.0f;

            for (int k = 0; k < COL_A; k++) {
                C[i * COL_B + j] += A[i * COL_A + k] * B[k * COL_B + j];
            }
        }
    }

    printf("GEMM_seq\tExecution time: %lfsec\n", (float)(clock() - start) / CLOCKS_PER_SEC);

}

void gemm_opencl(const float* A, const float* B, float* C, const int ROW_A, const int COL_A, const int ROW_B, const int COL_B) {
    cl_int err;

    // Platform ID
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);

    // Device ID
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    // Create Context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    // Create Command Queue
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    CHECK_ERROR(err);

    // Create Program Object
    size_t kernel_source_size;
    //char* kernel_source = get_source_code("kernel_2.cl", &kernel_source_size);
    char* kernel_source = get_source_code("kernel_3.cl", &kernel_source_size);
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
    CHECK_ERROR(err);

    // Build Program
    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    build_error(program, device, err);
    CHECK_ERROR(err);


    /*
     * 커널 생성은 여기서 진행해주세요.
     * ex)
     *  cl_kernel kernel = clCreateKernel(program, "gemm", &err);
     *  CHECK_ERROR(err);
     */

    cl_kernel kernel = clCreateKernel(program, "vec_mat", &err);
    CHECK_ERROR(err);


    // 버퍼 생성 : clCreateBuffer
    cl_mem buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * ROW_A * COL_A, NULL, &err);
    CHECK_ERROR(err);

    cl_mem buffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * ROW_B * COL_B, NULL, &err);
    CHECK_ERROR(err);

    cl_mem buffer_C = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * ROW_A * COL_A, NULL, &err);
    CHECK_ERROR(err);


    // 버퍼 등록 : clEnqueuWriteBuffer, clSetKernelArg
    err = clEnqueueWriteBuffer(queue, buffer_A, CL_TRUE, 0, sizeof(float) * ROW_A * COL_A, A, 0, NULL, NULL);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buffer_B, CL_TRUE, 0, sizeof(float) * ROW_B * COL_B, B, 0, NULL, NULL);
    CHECK_ERROR(err)

        // 커널 인자 설정
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_A); CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_B); CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_C); CHECK_ERROR(err);


    // 커널 실행 : clEnqueueNDRangeKernel
    size_t global_work_size[2] = { (size_t)1024, (size_t)1024 };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);


    // 결과 수집 : clEnqueueReadBuffer
    err = clEnqueueReadBuffer(queue, buffer_C, CL_TRUE, 0, sizeof(float) * ROW_A * COL_A, C, 0, NULL, NULL);
    CHECK_ERROR(err);



    clock_t start = clock();

    /*
     * 여기서부터 병렬 처리를 위한 나머지 호스트 코드를 작성하세요.
     */

    printf("GEMM_opencl\tExecution time: %lfsec\n", (float)(clock() - start) / CLOCKS_PER_SEC);



    clReleaseMemObject(buffer_A);
    clReleaseMemObject(buffer_B);
    clReleaseMemObject(buffer_C);
    err = clReleaseKernel(kernel);
    err = clReleaseProgram(program);
    err = clReleaseCommandQueue(queue);
    err = clReleaseContext(context);

    free(kernel_source);
}