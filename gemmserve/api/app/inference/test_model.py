import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from model import forward, load_labels, load_model, predict, softmax

OUTPUTS_DIR = Path(__file__).resolve().parents[3] / "scripts" / "outputs"
WEIGHTS_DIR = OUTPUTS_DIR / "weights"


# ---------------------------------------------------------------------------
# Unit tests (synthetic weights, no disk I/O)
# ---------------------------------------------------------------------------

def _make_layers(dims):
    """Create random numpy weight layers for given dimension list."""
    layers = []
    rng = np.random.default_rng(42)
    for in_d, out_d in zip(dims[:-1], dims[1:]):
        W = rng.standard_normal((out_d, in_d)).astype(np.float32)
        b = rng.standard_normal(out_d).astype(np.float32)
        layers.append((W, b))
    return layers


class TestForwardSynthetic:
    """Test forward pass logic with small synthetic weights."""

    def test_output_shape_single(self):
        layers = _make_layers([10, 8, 4])
        x = np.random.randn(10).astype(np.float32)
        out = forward(x, layers)
        assert out.shape == (4,)

    def test_output_shape_batch(self):
        layers = _make_layers([10, 8, 4])
        x = np.random.randn(5, 10).astype(np.float32)
        out = forward(x, layers)
        assert out.shape == (5, 4)

    def test_relu_kills_negatives(self):
        """Verify hidden activations go through ReLU (no negative outputs from hidden layers)."""
        # Single layer (no ReLU applied to final output)
        layers = _make_layers([4, 3])
        x = np.random.randn(4).astype(np.float32)
        out = forward(x, layers)
        # Final layer has no ReLU, so negatives are allowed
        assert out.shape == (3,)

        # Two layers: manually check that ReLU is applied to the hidden layer
        W0 = np.array([[1, 0], [0, -1]], dtype=np.float32)
        b0 = np.zeros(2, dtype=np.float32)
        W1 = np.eye(2, dtype=np.float32)
        b1 = np.zeros(2, dtype=np.float32)
        layers = [(W0, b0), (W1, b1)]
        x = np.array([1.0, 1.0], dtype=np.float32)
        # After W0: [1, -1], after ReLU: [1, 0], after W1: [1, 0]
        out = forward(x, layers)
        np.testing.assert_allclose(out, [1.0, 0.0], atol=1e-6)

    def test_bias_is_added(self):
        W = np.zeros((2, 3), dtype=np.float32)
        b = np.array([1.0, 2.0], dtype=np.float32)
        layers = [(W, b)]
        x = np.zeros(3, dtype=np.float32)
        out = forward(x, layers)
        np.testing.assert_allclose(out, [1.0, 2.0], atol=1e-6)


class TestSoftmax:
    def test_sums_to_one(self):
        logits = np.array([1.0, 2.0, 3.0])
        probs = softmax(logits)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)

    def test_batch_sums_to_one(self):
        logits = np.random.randn(4, 5).astype(np.float32)
        probs = softmax(logits)
        np.testing.assert_allclose(probs.sum(axis=-1), np.ones(4), atol=1e-6)

    def test_numerical_stability(self):
        logits = np.array([1000.0, 1001.0, 1002.0])
        probs = softmax(logits)
        assert np.all(np.isfinite(probs))
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Integration tests (require trained weights on disk)
# ---------------------------------------------------------------------------

needs_weights = pytest.mark.skipif(
    not (WEIGHTS_DIR / "manifest.json").exists(),
    reason="Exported weights not found; run export_weights.py first",
)

needs_pytorch_checkpoint = pytest.mark.skipif(
    not (OUTPUTS_DIR / "model.pt").exists(),
    reason="PyTorch checkpoint not found; run train_model.py first",
)


@needs_weights
class TestLoadModel:
    def test_loads_three_layers(self):
        layers = load_model()
        assert len(layers) == 3

    def test_layer_shapes(self):
        layers = load_model()
        assert layers[0][0].shape == (512, 50000)
        assert layers[0][1].shape == (512,)
        assert layers[1][0].shape == (256, 512)
        assert layers[1][1].shape == (256,)
        assert layers[2][0].shape == (20, 256)
        assert layers[2][1].shape == (20,)

    def test_dtypes(self):
        layers = load_model()
        for W, b in layers:
            assert W.dtype == np.float32
            assert b.dtype == np.float32


@needs_weights
class TestLoadLabels:
    def test_has_20_labels(self):
        labels = load_labels()
        assert len(labels) == 20

    def test_english_present(self):
        labels = load_labels()
        assert "en" in labels


@needs_weights
@needs_pytorch_checkpoint
class TestNumpyMatchesPyTorch:
    """Verify the numpy forward pass produces the same output as PyTorch."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        # Load numpy layers
        self.np_layers = load_model()

        # Rebuild PyTorch model from checkpoint
        ckpt = torch.load(OUTPUTS_DIR / "model.pt", map_location="cpu", weights_only=True)
        from sys import path as sys_path
        scripts_dir = str(Path(__file__).resolve().parents[3] / "scripts")
        if scripts_dir not in sys_path:
            sys_path.insert(0, scripts_dir)
        from train_model import LanguageMLP
        self.pt_model = LanguageMLP(
            ckpt["input_dim"], ckpt["hidden_sizes"], ckpt["num_classes"]
        )
        self.pt_model.load_state_dict(ckpt["model_state_dict"])
        self.pt_model.eval()

    def test_single_input(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(50000).astype(np.float32)

        np_out = forward(x, self.np_layers)
        with torch.no_grad():
            pt_out = self.pt_model(torch.from_numpy(x)).numpy()

        np.testing.assert_allclose(np_out, pt_out, rtol=1e-5, atol=1e-5)

    def test_batch_input(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((8, 50000)).astype(np.float32)

        np_out = forward(x, self.np_layers)
        with torch.no_grad():
            pt_out = self.pt_model(torch.from_numpy(x)).numpy()

        np.testing.assert_allclose(np_out, pt_out, rtol=1e-5, atol=1e-5)

    def test_argmax_agrees(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((16, 50000)).astype(np.float32)

        np_out = forward(x, self.np_layers)
        with torch.no_grad():
            pt_out = self.pt_model(torch.from_numpy(x)).numpy()

        np.testing.assert_array_equal(np.argmax(np_out, axis=1), np.argmax(pt_out, axis=1))
