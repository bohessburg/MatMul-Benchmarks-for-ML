U0 MatrixMultiply(F64 *A, F64 *B, F64 *C, I64 n)
{
  I64 i, j, k;
  F64 sum;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      sum = 0.0;
      for (k = 0; k < n; k++)
        sum += A[i * n + k] * B[k * n + j];
      C[i * n + j] = sum;
    }
}

U0 FillRandom(F64 *M, I64 n)
{
  I64 i;
  for (i = 0; i < n * n; i++)
    M[i] = ToF64(RandU32) / ToF64(U32_MAX);
}

I64 n = 512;

F64 *A = MAlloc(n * n * sizeof(F64));
F64 *B = MAlloc(n * n * sizeof(F64));
F64 *C = MAlloc(n * n * sizeof(F64));

FillRandom(A, n);
FillRandom(B, n);

"Multiplying %dx%d matrices...\n", n, n;

F64 start = tS;
MatrixMultiply(A, B, C, n);
F64 elapsed = tS - start;

F64 gflops = (2.0 * n * n * n) / (elapsed * 1e9);

F64 checksum = 0.0;
I64 i;
for (i = 0; i < n * n; i++)
  checksum += C[i];

"Naive (i,j,k):\n";
"  Elapsed:  %0.3f s\n", elapsed;
"  GFLOPS:   %0.3f\n", gflops;
"  Checksum: %0.6f\n", checksum;

Free(A);
Free(B);
Free(C);
