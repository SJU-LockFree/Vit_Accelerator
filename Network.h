#include <time.h>

#ifndef _Network_H
#define _Network_H

// 이미지 데이터 정보를 담을 구조체 정의
typedef struct {
    int n;      // 이미지 개수
    int c;      // 채널 수
    int h;      // 높이
    int w;      // 너비
    float* data; // 모든 이미지 데이터를 연속된 메모리 공간에 저장 (N x C x H x W)
} ImageData;

ImageData* load_image_data(const char* filename);

// Network 로드에 대한 로직
typedef struct {
    float* data;
    size_t size;
} Network;

double conv2d_t, pos_emb_t, ln1_t, mha_t, ln2_t, mlp_t;
double mlp_read, mlp_write, mlp_compute;
double mlp1, mlp2;
double attn1, attn2, attn3;
double attn2_1, attn2_2, attn2_3;
time_t start, end;
time_t start_t, end_t;
time_t start_mlp, end_mlp;
time_t start_attn, end_attn;
time_t start_2, end_2;

void load_weights(const char* directory, Network network[], int count);

#endif