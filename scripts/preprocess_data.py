from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np


def standardize_data(
    data: np.ndarray, mean=None, std=None, eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return (data - mean) / std, mean, std


def preprocess_and_save(raw_dir: str, out_dir: str, plot: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)

    filenames = sorted(f for f in os.listdir(raw_dir) if f.endswith("_data.npy"))

    if plot:
        import matplotlib.pyplot as plt

        n_files = len(filenames)
        n_cols = 3
        n_rows = (n_files + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
        axes = axes.flatten() if n_files > 1 else [axes]

    for idx, fname in enumerate(filenames):
        path = os.path.join(raw_dir, fname)
        data = np.load(path)

        data_std, _, _ = standardize_data(data)

        base, _ = os.path.splitext(fname)  # e.g. "lorenz_data"
        out_name = f"{base}_std.npy"  # -> "lorenz_data_std.npy"
        np.save(os.path.join(out_dir, out_name), data_std)

        if plot:
            ax = axes[idx]
            ax.plot(data_std[:, 0], label=f"{fname} (std)")
            ax.set_title(f"{fname} - First Dim (std)")
            ax.legend(loc="upper right")

    if plot:
        import matplotlib.pyplot as plt

        plt.tight_layout()
        plt.show()

    print(f"Processed data saved to '{out_dir}/'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="raw_data")
    ap.add_argument("--out_dir", type=str, default="processed_data")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    preprocess_and_save(args.raw_dir, args.out_dir, plot=args.plot)


if __name__ == "__main__":
    main()
