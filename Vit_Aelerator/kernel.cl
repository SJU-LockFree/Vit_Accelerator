

__kernel void vec_mat(__global int* A, __global int* B, __global int* C) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    int n = 1000;

    float sum = 0.0f;

    for (int k = 0; k < n; k++) {
        sum += A[row * n + k] * B[k * n + col];
    }

    C[row * n + col] = sum;
}