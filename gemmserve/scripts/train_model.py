#!/usr/bin/env python3
"""Train a language classification MLP on TF-IDF character n-gram features.

Uses the papluca/language-identification dataset (~90k samples, 20 languages).
Produces weight matrices suitable for GEMM benchmarking.
"""

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer


class LanguageMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int], num_classes: int):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train language classification MLP")
    p.add_argument("--max-features", type=int, default=50000,
                    help="TF-IDF vocabulary size (default: 50000)")
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 256],
                    help="Hidden layer sizes (default: 512 256)")
    p.add_argument("--epochs", type=int, default=10,
                    help="Number of training epochs (default: 10)")
    p.add_argument("--batch-size", type=int, default=256,
                    help="Mini-batch size (default: 256)")
    p.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate (default: 1e-3)")
    p.add_argument("--output-dir", type=str, default="outputs",
                    help="Directory for saved artifacts (default: outputs)")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load dataset ---
    print("Loading dataset (papluca/language-identification)...")
    ds = load_dataset("papluca/language-identification")
    train_texts = ds["train"]["text"]
    train_labels_str = ds["train"]["labels"]
    val_texts = ds["validation"]["text"]
    val_labels_str = ds["validation"]["labels"]

    # Build label mapping
    all_labels = sorted(set(train_labels_str))
    label2id = {l: i for i, l in enumerate(all_labels)}
    num_classes = len(all_labels)
    print(f"Languages ({num_classes}): {all_labels}")

    train_labels = np.array([label2id[l] for l in train_labels_str])
    val_labels = np.array([label2id[l] for l in val_labels_str])

    # --- TF-IDF feature extraction ---
    print(f"Fitting TF-IDF vectorizer (max_features={args.max_features})...")
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        max_features=args.max_features,
    )
    X_train = vectorizer.fit_transform(train_texts)  # sparse
    X_val = vectorizer.transform(val_texts)            # sparse
    input_dim = X_train.shape[1]
    print(f"Feature matrix: {X_train.shape[0]} samples × {input_dim} features (sparse)")

    # --- Build model ---
    model = LanguageMLP(input_dim, args.hidden_sizes, num_classes).to(device)
    print(f"\nModel architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # --- Training loop ---
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Shuffle training indices
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = math.ceil(n_train / args.batch_size)

        for b in range(n_batches):
            start = b * args.batch_size
            end = min(start + args.batch_size, n_train)
            idx = perm[start:end]

            # Convert sparse batch to dense tensor
            batch_x = X_train[idx]
            if issparse(batch_x):
                batch_x = batch_x.toarray()
            batch_x = torch.tensor(batch_x, dtype=torch.float32, device=device)
            batch_y = torch.tensor(train_labels[idx], dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches

        # --- Validation ---
        model.eval()
        correct = 0
        with torch.no_grad():
            for b in range(math.ceil(n_val / args.batch_size)):
                start = b * args.batch_size
                end = min(start + args.batch_size, n_val)

                batch_x = X_val[start:end]
                if issparse(batch_x):
                    batch_x = batch_x.toarray()
                batch_x = torch.tensor(batch_x, dtype=torch.float32, device=device)
                batch_y = torch.tensor(val_labels[start:end], dtype=torch.long, device=device)

                logits = model(batch_x)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_y).sum().item()

        val_acc = correct / n_val
        print(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    # --- Save artifacts ---
    checkpoint_path = output_dir / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_sizes": args.hidden_sizes,
        "num_classes": num_classes,
    }, checkpoint_path)
    print(f"\nSaved checkpoint: {checkpoint_path}")

    vectorizer_path = output_dir / "vectorizer.pkl"
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Saved vectorizer: {vectorizer_path}")

    labels_path = output_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump({"labels": all_labels, "label2id": label2id}, f, indent=2)
    print(f"Saved labels:     {labels_path}")


if __name__ == "__main__":
    main()
