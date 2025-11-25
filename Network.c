#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "dirent.h"
#include <math.h>  // roundf 함수를 사용하기 위해 추가

// 이미지 데이터 정보를 담을 구조체 정의
typedef struct {
    int n;      // 이미지 개수
    int c;      // 채널 수
    int h;      // 높이
    int w;      // 너비
    float* data; // 모든 이미지 데이터를 연속된 메모리 공간에 저장 (N x C x H x W)
} ImageData;

typedef struct {
    float* data;
    size_t size;   // float 원소 개수
} Network;

// input.bin 파일의 헤더는 4개의 int32: (n, c, h, w)
// 이후 float32 데이터가 연속해서 저장되어 있다고 가정합니다.
ImageData* load_image_data(const char* filename) {
    FILE* f = NULL;
    errno_t err = fopen_s(&f, filename, "rb");
    if (err != 0) {
        // 에러 처리: 파일 열기에 실패한 경우
    }
    if (f == NULL) {
        perror("파일 열기 실패");
        return NULL;
    }

    // 헤더 읽기: n, c, h, w
    int header[4];
    if (fread(header, sizeof(int), 4, f) != 4) {
        perror("헤더 읽기 실패");
        fclose(f);
        return NULL;
    }

    int n = header[0];
    int c = header[1];
    int h = header[2];
    int w = header[3];
    int image_size = c * h * w;
    //printf("%d * %d * %d = %d!!\n", c, h, w, image_size);
    int total_elements = n * image_size;

    // 모든 이미지 데이터를 한 번에 읽어 들일 임시 버퍼 할당
    float* all_data = (float*)malloc(total_elements * sizeof(float));
    if (all_data == NULL) {
        perror("전체 데이터 버퍼 메모리 할당 실패");
        fclose(f);
        return NULL;
    }
    if (fread(all_data, sizeof(float), total_elements, f) != total_elements) {
        perror("이미지 데이터 읽기 실패");
        free(all_data);
        fclose(f);
        return NULL;
    }
    fclose(f);

    // 이미지 개수만큼의 ImageData 배열 할당
    ImageData* images = (ImageData*)malloc(n * sizeof(ImageData));
    if (images == NULL) {
        perror("ImageData 배열 메모리 할당 실패");
        free(all_data);
        return NULL;
    }

    // 각 이미지별로 메타정보를 설정하고, 데이터는 별도의 메모리 영역에 복사
    for (int i = 0; i < n; i++) {
        images[i].n = n;  // 전체 이미지 개수를 저장 (편의를 위해 각 구조체에 동일하게 저장)
        images[i].c = c;
        images[i].h = h;
        images[i].w = w;
        images[i].data = (float*)malloc(image_size * sizeof(float));
        if (images[i].data == NULL) {
            perror("개별 이미지 데이터 메모리 할당 실패");
            // 에러 발생 시 이전에 할당한 메모리 해제
            for (int j = 0; j < i; j++) {
                free(images[j].data);
            }
            free(images);
            free(all_data);
            return NULL;
        }
        // 모든 이미지 데이터가 연속으로 저장되어 있으므로, i번째 이미지 데이터를 복사
        memcpy(images[i].data, all_data + i * image_size, image_size * sizeof(float));
    }

    free(all_data);
    return images;
}

static int parse_index_from_filename(const char* filename) {
    // filename이 "Weight_"로 시작하는지 확인
    if (strncmp(filename, "Weight_", 7) != 0) {
        return -1;
    }
    // "Weight_" 이후부터 '_' 문자가 나올 때까지의 문자열 추출
    const char* start = filename + 7;
    const char* end = strchr(start, '_');
    if (!end) {
        return -1;
    }
    size_t len = end - start;
    char index_str[16] = { 0 };
    if (len >= sizeof(index_str))
        len = sizeof(index_str) - 1;
    strncpy_s(index_str, sizeof(index_str), start, len);
    index_str[len] = '\0';
    return atoi(index_str);
}

void load_weights(const char* directory, Network network[], int count) {
    DIR* dir = opendir(directory);
    if (!dir) {
        perror("디렉토리 열기 실패");
        exit(EXIT_FAILURE);
    }

    // network 배열 초기화
    for (int i = 0; i < count; i++) {
        network[i].data = NULL;
        network[i].size = 0;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        // 파일명이 "Weight_"로 시작하는지 확인
        if (strncmp(entry->d_name, "Weight_", 7) != 0)
            continue;
        // 확장자가 ".bin"인지 확인
        const char* ext = strrchr(entry->d_name, '.');
        if (!ext || strcmp(ext, ".bin") != 0)
            continue;

        int idx = parse_index_from_filename(entry->d_name);
        if (idx < 0 || idx >= count)
            continue;

        // 전체 경로 생성: 예) "./Network/Weight_96_encoder_layers_encoder_layer_7_mlp_0_weight.bin"
        char filepath[512];
        snprintf(filepath, sizeof(filepath), "%s/%s", directory, entry->d_name);

        // 파일을 바이너리 읽기 모드로 열기
        FILE* fp = NULL;
        errno_t err = fopen_s(&fp, filepath, "rb");
        if (err != 0) {
            // 에러 처리: 파일 열기에 실패한 경우
        }

        // 파일 크기 확인 (바이트 단위)
        fseek(fp, 0, SEEK_END);
        long file_size = ftell(fp);
        rewind(fp);
        if (file_size < 0) {
            fclose(fp);
            continue;
        }
        // 파일이 float 배열이라고 가정하므로 float 원소 개수 계산
        size_t num_floats = file_size / sizeof(float);

        // float 배열을 저장할 메모리 할당
        float* buffer = (float*)malloc(file_size);
        if (!buffer) {
            perror("메모리 할당 실패");
            fclose(fp);
            exit(EXIT_FAILURE);
        }
        size_t read_size = fread(buffer, sizeof(float), num_floats, fp);
        if (read_size != num_floats) {
            perror("파일 읽기 오류");
            free(buffer);
            fclose(fp);
            continue;
        }
        fclose(fp);

        // 각 float 값을 소수점 6자리까지 반올림
        for (size_t i = 0; i < num_floats; i++) {
            buffer[i] = roundf(buffer[i] * 1000000.0f) / 1000000.0f;
        }

        // 인덱스에 해당하는 위치에 데이터와 크기를 저장
        network[idx].data = buffer;
        network[idx].size = num_floats;
    }
    closedir(dir);
}
