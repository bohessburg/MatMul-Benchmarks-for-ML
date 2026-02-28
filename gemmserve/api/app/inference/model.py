import json
import numpy as np
from pathlib import Path


WEIGHTS_DIR = Path(__file__).resolve().parents[3] / "scripts" / "outputs" / "weights"
OUTPUTS_DIR = Path(__file__).resolve().parents[3] / "scripts" / "outputs"


def load_model(weights_dir: Path = WEIGHTS_DIR):
    """Load weight matrices and bias vectors from exported binaries."""
    with open(weights_dir / "manifest.json") as f:
        manifest = json.load(f)

    layers = []
    for layer in manifest["layers"]:
        W = np.fromfile(weights_dir / layer["weight_file"], dtype=np.float32)
        W = W.reshape(layer["out_features"], layer["in_features"])
        b = np.fromfile(weights_dir / layer["bias_file"], dtype=np.float32)
        layers.append((W, b))
    return layers


def load_labels(outputs_dir: Path = OUTPUTS_DIR):
    """Load the label list from labels.json."""
    with open(outputs_dir / "labels.json") as f:
        data = json.load(f)
    return data["labels"]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def forward(x: np.ndarray, layers: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    MLP forward pass: Linear -> ReLU -> Linear -> ReLU -> Linear.

    x: input array of shape (batch, 50000) or (50000,)
    Returns: logits of shape (batch, 20) or (20,)
    """
    for W, b in layers[:-1]:
        x = x @ W.T + b
        x = np.maximum(x, 0)
    W, b = layers[-1]
    x = x @ W.T + b
    return x


def predict(x: np.ndarray, layers, labels):
    """Run forward pass and return the predicted language label."""
    logits = forward(x, layers)
    idx = np.argmax(logits, axis=-1)
    return labels[idx]
