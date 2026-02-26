#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void (*mul_fn)(const double *, const double *, double *, int);

typedef struct {
    const char *name;
    mul_fn      func;
} algorithm;

void matrix_multiply(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void matrix_multiply_ikj(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double a_ik = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

#define TILE_SIZE 64

void matrix_multiply_tiled(const double *A, const double *B, double *C, int n) {
    memset(C, 0, n * n * sizeof(double));
    for (int i0 = 0; i0 < n; i0 += TILE_SIZE) {
        for (int k0 = 0; k0 < n; k0 += TILE_SIZE) {
            for (int j0 = 0; j0 < n; j0 += TILE_SIZE) {
                int i_end = i0 + TILE_SIZE < n ? i0 + TILE_SIZE : n;
                int k_end = k0 + TILE_SIZE < n ? k0 + TILE_SIZE : n;
                int j_end = j0 + TILE_SIZE < n ? j0 + TILE_SIZE : n;
                for (int i = i0; i < i_end; i++) {
                    for (int k = k0; k < k_end; k++) {
                        double a_ik = A[i * n + k];
                        for (int j = j0; j < j_end; j++) {
                            C[i * n + j] += a_ik * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

void fill_random(double *M, int n) {
    for (int i = 0; i < n * n; i++) {
        M[i] = (double)rand() / RAND_MAX;
    }
}

int main(int argc, char *argv[]) {
    algorithm algos[] = {
        {"naive_ijk",    matrix_multiply},
        {"reordered_ikj", matrix_multiply_ikj},
        {"tiled_64",      matrix_multiply_tiled},
    };
    int num_algos = sizeof(algos) / sizeof(algos[0]);

    int n = 1024;
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    /* Pick algorithm */
    printf("Select algorithm:\n");
    for (int i = 0; i < num_algos; i++) {
        printf("  %d) %s\n", i + 1, algos[i].name);
    }
    printf("  0) Run all\n");
    printf("Choice: ");

    int choice;
    if (scanf("%d", &choice) != 1 || choice < 0 || choice > num_algos) {
        fprintf(stderr, "Invalid choice\n");
        return 1;
    }

    /* Allocate and fill matrices */
    double *A = malloc(n * n * sizeof(double));
    double *B = malloc(n * n * sizeof(double));
    double *C = malloc(n * n * sizeof(double));
    if (!A || !B || !C) {
        fprintf(stderr, "Failed to allocate matrices (%dx%d)\n", n, n);
        return 1;
    }

    srand(42);
    fill_random(A, n);
    fill_random(B, n);

    printf("\nMultiplying %dx%d matrices...\n\n", n, n);

    /* Benchmark selected algorithm(s) */
    int start_idx = (choice == 0) ? 0 : choice - 1;
    int end_idx   = (choice == 0) ? num_algos : choice;

    for (int a = start_idx; a < end_idx; a++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        algos[a].func(A, B, C, n);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        double gflops = (2.0 * n * n * n) / (elapsed * 1e9);
        double checksum = 0.0;
        for (int i = 0; i < n * n; i++) checksum += C[i];

        printf("%s:\n", algos[a].name);
        printf("  Elapsed:  %.3f s\n", elapsed);
        printf("  GFLOPS:   %.3f\n", gflops);
        printf("  Checksum: %f\n\n", checksum);
    }

    free(A);
    free(B);
    free(C);
    return 0;
}
