import ctypes
import platform
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np

MatMulFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

KERNEL_DIR = Path(__file__).resolve().parents[3] / "kernel" / "cpp"
SRC_PATH = KERNEL_DIR / "gemm.cpp"
LIB_NAME = "libgemm.dylib" if platform.system() == "Darwin" else "libgemm.so"
LIB_PATH = KERNEL_DIR / LIB_NAME

_lib: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    """Compile (if needed) and load the C++ shared library."""
    global _lib
    if _lib is not None:
        return _lib

    if not LIB_PATH.exists() or SRC_PATH.stat().st_mtime > LIB_PATH.stat().st_mtime:
        subprocess.check_call(
            ["c++", "-O2", "-shared", "-fPIC", "-o", str(LIB_PATH), str(SRC_PATH)]
        )

    _lib = ctypes.CDLL(str(LIB_PATH))

    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_int = ctypes.c_int
    arg_types = [c_float_p, c_float_p, c_float_p, c_int, c_int, c_int]

    for name in ("matmul_basic_ijk", "matmul_reordered_ikj", "matmul_tiled"):
        fn = getattr(_lib, name)
        fn.argtypes = arg_types
        fn.restype = None

    return _lib


def _cpp_matmul(fn_name: str, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Call a C++ kernel function, returning the result as a numpy array."""
    lib = _load_lib()
    fn = getattr(lib, fn_name)

    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dimensions must match: {K} != {K2}"

    C = np.zeros((M, N), dtype=np.float32)

    fn(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
    )
    return C


def numpy_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B


def naive_ijk(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return _cpp_matmul("matmul_basic_ijk", A, B)


def optimized_ikj(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return _cpp_matmul("matmul_reordered_ikj", A, B)


def tiled(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return _cpp_matmul("matmul_tiled", A, B)


# Registry: add new kernels here
KERNELS: dict[str, MatMulFn] = {
    "numpy": numpy_matmul,
    "naive_ijk": naive_ijk,
    "optimized_ikj": optimized_ikj,
    "tiled": tiled,
}


def get_kernel(name: str) -> MatMulFn:
    """Look up a kernel by name. Raises KeyError if not found."""
    return KERNELS[name]
