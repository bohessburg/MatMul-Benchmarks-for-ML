extern "C" {

//Naive O(n^3) matrix multiplication with no cache awareness
//A is MxK, B is KxN, C is MxN
void matmul_basic_ijk(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

//Optimized basic O(n^3) algorithm with inner loop and middle loop swapped.
//This results in inner loop accessing contiguous memory segments, so more cached memory is used
//A is MxK, B is KxN, C is MxN
void matmul_reordered_ikj(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A[i * K + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

//Tiled matrix multiplication
//Divides matrix into discrete "tiles" to improve cache locality.
//Each tile fits in L1/L2 cache so accesses within a tile are fast.
//A is MxK, B is KxN, C is MxN
#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i0 = 0; i0 < M; i0 += TILE_SIZE) {
        for (int j0 = 0; j0 < N; j0 += TILE_SIZE) {
            for (int k0 = 0; k0 < K; k0 += TILE_SIZE) {
                // Compute bounds for this tile, handling edge tiles that may be smaller
                int i_end = (i0 + TILE_SIZE < M) ? i0 + TILE_SIZE : M;
                int j_end = (j0 + TILE_SIZE < N) ? j0 + TILE_SIZE : N;
                int k_end = (k0 + TILE_SIZE < K) ? k0 + TILE_SIZE : K;

                for (int i = i0; i < i_end; i++) {
                    for (int k = k0; k < k_end; k++) {
                        float a_ik = A[i * K + k];
                        for (int j = j0; j < j_end; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"
