from __future__ import annotations

import os

import numpy as np


def load_data_files_whole(processed_dir: str, input_dim: int, output_dim: int):
    """
    Loads *_data_std.npy files and returns list of (fname, file_in, file_tg)
    where file_in = arr[:-1, :input_dim], file_tg = arr[1:, :output_dim].
    """
    data_files = sorted(f for f in os.listdir(processed_dir) if f.endswith("_data_std.npy"))
    file_list = []

    for fname in data_files:
        arr = np.load(os.path.join(processed_dir, fname))
        if arr.shape[0] < 2:
            continue

        max_cols = max(input_dim, output_dim)
        if arr.shape[1] < max_cols:
            pad_cols = max_cols - arr.shape[1]
            arr = np.concatenate([arr, np.zeros((arr.shape[0], pad_cols))], axis=1)

        file_in = arr[:-1, :input_dim]
        file_tg = arr[1:, :output_dim]
        file_list.append((fname, file_in, file_tg))

    return file_list
