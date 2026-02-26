//Naive O(n^3) matrix multiplication with no cache awareness
void matmul_basic_ijk(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++) { // for each row of A
        for (int j = 0; j < n; j++) { // for each column of B
            float sum = 0.0f;
            for (int k = 0; k < n; k++) { // take dot product
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

//Optimized basic O(n^3) algorithm with inner loop and middle loop swapped.
//This results in inner loop accessing contiguous memory segments, so more cached memory is used
void matmul_reordered_ikj(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++) { // for each row of A
        for (int k = 0; k < n; k++) { // take dot product with:
            float a_ik = A[i * n + k]; // store value in A matrix into register for faster access
            for (int j = 0; j < n; j++) {
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}