#pragma once
#include <CL/cl.h>
#include "Network.h" // ImageData 정의 필요

void ViT_opencl(ImageData* image, cl_mem* d_networks, float** probabilities, cl_context context, cl_command_queue queue, cl_program program, cl_device_id device);