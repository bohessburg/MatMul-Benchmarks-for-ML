#!/usr/bin/env python3
"""Export trained MLP weights as raw float32 binaries for C++ GEMM kernels.

Each nn.Linear layer produces:
  - layerN_weight.bin  (float32, row-major, shape [out_features, in_features])
  - layerN_bias.bin    (float32, shape [out_features])

A manifest.json describes dimensions and filenames so C++ code knows what to load.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export MLP weights to raw float32 binaries")
    p.add_argument("--checkpoint", type=str, default="outputs/model.pt",
                    help="Path to model checkpoint (default: outputs/model.pt)")
    p.add_argument("--output-dir", type=str, default="outputs/weights",
                    help="Directory for weight binaries (default: outputs/weights)")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    # Rebuild model to extract layers in order
    from train_model import LanguageMLP
    model = LanguageMLP(
        input_dim=ckpt["input_dim"],
        hidden_sizes=ckpt["hidden_sizes"],
        num_classes=ckpt["num_classes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Find all Linear layers
    linear_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]

    manifest = {"layers": []}

    for i, (name, layer) in enumerate(linear_layers):
        weight = layer.weight.detach().numpy().astype(np.float32)  # [out, in]
        bias = layer.bias.detach().numpy().astype(np.float32)      # [out]

        w_file = f"layer{i}_weight.bin"
        b_file = f"layer{i}_bias.bin"

        # Save as contiguous row-major float32
        weight.tofile(output_dir / w_file)
        bias.tofile(output_dir / b_file)

        layer_info = {
            "index": i,
            "name": name,
            "weight_file": w_file,
            "bias_file": b_file,
            "out_features": int(layer.out_features),
            "in_features": int(layer.in_features),
            "weight_shape": list(weight.shape),
            "dtype": "float32",
            "layout": "row_major",
        }
        manifest["layers"].append(layer_info)

        size_mb = weight.nbytes / (1024 * 1024)
        print(f"Layer {i} ({name}): weight {list(weight.shape)} ({size_mb:.1f} MB), bias {list(bias.shape)}")

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest: {manifest_path}")
    print(f"Exported {len(linear_layers)} layers to {output_dir}/")


if __name__ == "__main__":
    main()
