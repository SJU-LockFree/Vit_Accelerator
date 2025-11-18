#pragma once

#ifndef __GEMM__
#define __GEMM__

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

#include <CL/cl.h>
#include <stdlib.h>

char* get_source_code(const char* file_name, size_t* len);
void build_error(cl_program program, cl_device_id device, cl_int err);
void gemm_seq(const float* A, const float* B, float* C,
    const int ROW_A, const int COL_A, const int ROW_B, const int COL_B);
void gemm_opencl(const float* A, const float* B, float* C,
    const int ROW_A, const int COL_A, const int ROW_B, const int COL_B);

#endif