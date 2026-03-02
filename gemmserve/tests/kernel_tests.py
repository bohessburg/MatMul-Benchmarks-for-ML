import ctypes
import subprocess
import platform
import os

import numpy as np
import pytest

KERNEL_DIR = os.path.join(os.path.dirname(__file__), "..", "kernel", "cpp")
SRC_PATH = os.path.join(KERNEL_DIR, "gemm.cpp")

if platform.system() == "Darwin":
    LIB_NAME = "libgemm.dylib"
else:
    LIB_NAME = "libgemm.so"

LIB_PATH = os.path.join(KERNEL_DIR, LIB_NAME)


@pytest.fixture(scope="session", autouse=True)
def compile_kernel():
    """Compile the C++ kernel into a shared library once per test session."""
    subprocess.check_call(
        ["c++", "-O2", "-shared", "-fPIC", "-o", LIB_PATH, SRC_PATH]
    )
    yield
    if os.path.exists(LIB_PATH):
        os.remove(LIB_PATH)


@pytest.fixture(scope="session")
def lib(compile_kernel):
    """Load the compiled shared library and set up function signatures."""
    gemm = ctypes.CDLL(LIB_PATH)

    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_int = ctypes.c_int
    arg_types = [c_float_p, c_float_p, c_float_p, c_int, c_int, c_int]

    for fn_name in ("matmul_basic_ijk", "matmul_reordered_ikj", "matmul_tiled"):
        fn = getattr(gemm, fn_name)
        fn.argtypes = arg_types
        fn.restype = None

    return gemm


def _run_matmul(lib, fn_name, A, B):
    """Call a C kernel function and return the result as a numpy array."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    A_c = A.astype(np.float32, copy=True)
    B_c = B.astype(np.float32, copy=True)
    C_c = np.zeros((M, N), dtype=np.float32)

    fn = getattr(lib, fn_name)
    fn(
        A_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
    )
    return C_c


KERNELS = ["matmul_basic_ijk", "matmul_reordered_ikj", "matmul_tiled"]


# ---- Square matrix tests ----

@pytest.mark.parametrize("fn_name", KERNELS)
def test_identity_matrix(lib, fn_name):
    """A @ I = A"""
    n = 64
    A = np.random.randn(n, n).astype(np.float32)
    I = np.eye(n, dtype=np.float32)
    C = _run_matmul(lib, fn_name, A, I)
    np.testing.assert_allclose(C, A, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("fn_name", KERNELS)
def test_zero_matrix(lib, fn_name):
    """A @ 0 = 0"""
    n = 64
    A = np.random.randn(n, n).astype(np.float32)
    Z = np.zeros((n, n), dtype=np.float32)
    C = _run_matmul(lib, fn_name, A, Z)
    np.testing.assert_array_equal(C, Z)


@pytest.mark.parametrize("fn_name", KERNELS)
def test_square_random(lib, fn_name):
    """Random square matrices match numpy."""
    n = 128
    A = np.random.randn(n, n).astype(np.float32)
    B = np.random.randn(n, n).astype(np.float32)
    C = _run_matmul(lib, fn_name, A, B)
    expected = A @ B
    np.testing.assert_allclose(C, expected, rtol=1e-4, atol=1e-4)


# ---- Rectangular matrix tests ----

@pytest.mark.parametrize("fn_name", KERNELS)
def test_rectangular_wide_result(lib, fn_name):
    """(M×K) @ (K×N) where N > M."""
    M, K, N = 32, 64, 128
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = _run_matmul(lib, fn_name, A, B)
    expected = A @ B
    np.testing.assert_allclose(C, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("fn_name", KERNELS)
def test_rectangular_tall_result(lib, fn_name):
    """(M×K) @ (K×N) where M > N."""
    M, K, N = 128, 64, 32
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = _run_matmul(lib, fn_name, A, B)
    expected = A @ B
    np.testing.assert_allclose(C, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("fn_name", KERNELS)
def test_rectangular_non_tile_aligned(lib, fn_name):
    """Dimensions that don't divide evenly by tile size (32)."""
    M, K, N = 37, 53, 41
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = _run_matmul(lib, fn_name, A, B)
    expected = A @ B
    np.testing.assert_allclose(C, expected, rtol=1e-4, atol=1e-4)


# ---- Edge cases ----

@pytest.mark.parametrize("fn_name", KERNELS)
def test_single_element(lib, fn_name):
    """1×1 matrix multiplication."""
    A = np.array([[3.0]], dtype=np.float32)
    B = np.array([[7.0]], dtype=np.float32)
    C = _run_matmul(lib, fn_name, A, B)
    np.testing.assert_allclose(C, [[21.0]])


@pytest.mark.parametrize("fn_name", KERNELS)
def test_vector_outer_product(lib, fn_name):
    """Column vector × row vector = outer product."""
    col = np.array([[1], [2], [3]], dtype=np.float32)  # 3×1
    row = np.array([[4, 5, 6]], dtype=np.float32)       # 1×3
    C = _run_matmul(lib, fn_name, col, row)
    expected = col @ row
    np.testing.assert_allclose(C, expected)


@pytest.mark.parametrize("fn_name", KERNELS)
def test_vector_dot_product(lib, fn_name):
    """Row vector × column vector = dot product (1×1 result)."""
    row = np.array([[1, 2, 3]], dtype=np.float32)  # 1×3
    col = np.array([[4], [5], [6]], dtype=np.float32)  # 3×1
    C = _run_matmul(lib, fn_name, row, col)
    expected = row @ col
    np.testing.assert_allclose(C, expected)


# ---- Cross-kernel consistency ----

def test_all_kernels_agree(lib):
    """All three kernels produce the same result on the same input."""
    M, K, N = 50, 70, 60
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    results = [_run_matmul(lib, fn, A, B) for fn in KERNELS]
    for i in range(1, len(results)):
        np.testing.assert_allclose(results[0], results[i], rtol=1e-5, atol=1e-5)
