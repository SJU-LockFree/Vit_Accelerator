#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define BUFFER_SIZE 1024
#define IMAGE_COUNT 100

// 문자열에서 index(label)와 probability 추출
int parse_line(const char* line, int* label, float* prob) {
    // 포맷: [0] label: 65 / prob: 0.919345)
    return sscanf(line, "[%*d] label: %d / prob: %f)", label, prob);
}

// 개행 제거
static void trim_newline(char* str) {
    size_t len = strlen(str);
    if (len > 0 && str[len - 1] == '\n')
        str[len - 1] = '\0';
}

int comparator(void) {
    FILE* fp_result = NULL;
    FILE* fp_answer = NULL;
    errno_t err1 = fopen_s(&fp_result, "./Data/opencl_result.txt", "r");
    errno_t err2 = fopen_s(&fp_answer, "./Data/answer_result.txt", "r");

    if (err1 != 0 || fp_result == NULL) {
        fprintf(stderr, "Error: Cannot open ./Data/opencl_result.txt\n");
        return 1;
    }
    if (err2 != 0 || fp_answer == NULL) {
        fprintf(stderr, "Error: Cannot open ./Data/answer_result.txt\n");
        fclose(fp_result);
        return 1;
    }

    char line_result[BUFFER_SIZE];
    char line_answer[BUFFER_SIZE];
    int errors = 0;

    for (int line_number = 0; line_number < IMAGE_COUNT; ++line_number) {
        if (fgets(line_result, BUFFER_SIZE, fp_result) == NULL ||
            fgets(line_answer, BUFFER_SIZE, fp_answer) == NULL) {
            fprintf(stderr, "Line %d: 파일의 줄 수가 충분하지 않습니다.\n", line_number);
            errors++;
            break;
        }

        trim_newline(line_result);
        trim_newline(line_answer);

        int label_r, label_a;
        float prob_r, prob_a;

        if (parse_line(line_result, &label_r, &prob_r) != 2 ||
            parse_line(line_answer, &label_a, &prob_a) != 2) {
            fprintf(stderr, "Line %d: 파싱 오류 발생\n", line_number);
            errors++;
            continue;
        }

        if (label_r != label_a) {
            fprintf(stderr, "Line %d: Label mismatch (Result: %d, Answer: %d)\n",
                line_number, label_r, label_a);
            errors++;
        }

        if (fabs(prob_r - prob_a) > 0.01f) {
            fprintf(stderr, "Line %d: Probability mismatch (Result: %.6f, Answer: %.6f)\n",
                line_number, prob_r, prob_a);
            errors++;
        }
    }

    fclose(fp_result);
    fclose(fp_answer);
    return errors;
}
