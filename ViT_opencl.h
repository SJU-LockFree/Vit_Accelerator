// ViT_opencl.h
#ifndef VIT_OPENCL_H
#define VIT_OPENCL_H

#include <CL/cl.h>
#include "Network.h" // ImageData, Network 구조체 정의 필요

// 메인 함수에서 호출하는 함수 프로토타입
void ViT_opencl(ImageData* image, cl_mem* d_networks, float** probabilities,
    cl_context context, cl_command_queue queue, cl_program program, cl_device_id device);

#endif