"""
Benchmark harness for GEMM kernels.

Measures each kernel across a range of matrix sizes (square and MLP-relevant)
and writes structured results to bench.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add the API source so we can import kernels directly
sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".." / "api"))

from app.inference.kernels import KERNELS

# ---------------------------------------------------------------------------
# 1. Define matrix sizes to benchmark
# ---------------------------------------------------------------------------

# Square matrices: standard sizes that reveal scaling behavior.
# Small sizes (128) show function-call overhead; large sizes (2048) show
# cache/memory effects and where tiling really pays off.
SQUARE_SIZES = [128, 256, 512, 1024, 2048]

# MLP-relevant shapes: (M, K, N) tuples matching your model's actual matmuls.
# During forward pass, each layer computes x @ W.T, so:
#   Layer 0: (1, 50000) @ (50000, 512)  ->  M=1, K=50000, N=512
#   Layer 1: (1, 512)   @ (512, 256)    ->  M=1, K=512,   N=256
#   Layer 2: (1, 256)   @ (256, 20)     ->  M=1, K=256,   N=20
MLP_SHAPES = [
    (1, 50000, 512),  # layer 0
    (1, 512, 256),    # layer 1
    (1, 256, 20),     # layer 2
]

# How many times to repeat each measurement. More reps = more stable median.
NUM_REPS = 7


# ---------------------------------------------------------------------------
# 2. Timing helper
# ---------------------------------------------------------------------------

def bench_one(kernel_fn, M: int, K: int, N: int, reps: int = NUM_REPS) -> dict:
    """
    Time a single kernel on random (M,K) x (K,N) matrices.

    Returns a dict with shape info, median/min/max latency, and GFLOPS.
    """
    # Generate random float32 matrices (the values don't matter for timing)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    # Warmup: one untimed call so any lazy init (C++ compile) happens here
    kernel_fn(A, B)

    # Timed runs
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        kernel_fn(A, B)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    median_s = float(np.median(times))

    # FLOPS for matmul: each of the M*N output elements needs K multiply-adds
    # = 2 * M * N * K floating-point operations
    flops = 2.0 * M * N * K
    gflops = (flops / median_s) / 1e9 if median_s > 0 else 0.0

    return {
        "M": M,
        "K": K,
        "N": N,
        "reps": reps,
        "median_ms": round(median_s * 1000, 3),
        "min_ms": round(min(times) * 1000, 3),
        "max_ms": round(max(times) * 1000, 3),
        "gflops": round(gflops, 3),
    }


# ---------------------------------------------------------------------------
# 3. Main: run all benchmarks and save to bench.json
# ---------------------------------------------------------------------------

def main():
    output_path = Path(__file__).resolve().parent / "bench.json"

    # Build the list of (label, M, K, N) test cases
    cases = []
    for sz in SQUARE_SIZES:
        cases.append((f"square_{sz}", sz, sz, sz))
    for M, K, N in MLP_SHAPES:
        cases.append((f"mlp_{M}x{K}x{N}", M, K, N))

    results = {}

    for kernel_name, kernel_fn in KERNELS.items():
        print(f"\n{'='*50}")
        print(f"Kernel: {kernel_name}")
        print(f"{'='*50}")
        results[kernel_name] = {}

        for label, M, K, N in cases:
            print(f"  {label:>20s}  ({M}x{K}) @ ({K}x{N}) ... ", end="", flush=True)

            try:
                stats = bench_one(kernel_fn, M, K, N)
                results[kernel_name][label] = stats
                print(f"{stats['median_ms']:>10.3f} ms   {stats['gflops']:>8.3f} GFLOPS")
            except Exception as e:
                print(f"FAILED: {e}")
                results[kernel_name][label] = {"error": str(e)}

    # Write results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
