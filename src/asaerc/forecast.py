"""Closed-loop (autonomous) forecasting and valid prediction time."""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def closed_loop_forecast(
    pde_res, readout, warmup_inputs: np.ndarray, steps: int, device, delta: bool = False
) -> np.ndarray:
    """
    Drives the reservoir with ``warmup_inputs`` (teacher forcing), then feeds the readout's
    prediction back as the next input for ``steps`` steps.

    ``prediction[0]`` is the estimate of the value following ``warmup_inputs[-1]``. The readout's
    output is used as the next input, so ``output_dim`` must cover the input dimensions; extra
    outputs are ignored and missing inputs are padded with zeros.

    ``delta=True``: the readout predicts the change y_{t+1} - x_t (``target_mode: delta``).

    Returns:
      predictions [steps, output_dim]
    """
    pde_res.eval()
    readout.eval()
    pde_res.reset_state(batch_size=1, device=str(device))
    input_dim = pde_res.input_size

    def step(u: np.ndarray) -> np.ndarray:
        u_t = torch.as_tensor(np.asarray(u, dtype=np.float32), device=device).unsqueeze(0)
        meas = pde_res(u_t)
        out = readout(meas, pde_res.state.unsqueeze(0))
        return out[0].float().cpu().numpy()

    warmup_inputs = np.asarray(warmup_inputs, dtype=np.float32)
    for u in warmup_inputs[:-1]:
        step(u)

    next_input = warmup_inputs[-1]
    predictions = []
    for _ in range(steps):
        out = step(next_input)
        if delta:
            offset = np.zeros_like(out)
            k = min(out.shape[0], input_dim)
            offset[:k] = next_input[:k]
            out = out + offset
        predictions.append(out)
        next_input = np.zeros(input_dim, dtype=np.float32)
        n = min(input_dim, out.shape[0])
        next_input[:n] = out[:n]
    return np.asarray(predictions)


def normalized_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-step ||pred - true|| / sqrt(<||true||^2>)."""
    prediction = np.atleast_2d(prediction)
    target = np.atleast_2d(target)
    n = min(len(prediction), len(target))
    scale = np.sqrt(np.mean(np.sum(target[:n] ** 2, axis=1)))
    return np.linalg.norm(prediction[:n] - target[:n], axis=1) / (scale if scale > 0 else 1.0)


def valid_prediction_time(
    prediction: np.ndarray, target: np.ndarray, threshold: float = 0.4
) -> int:
    """Number of steps until the normalized error first exceeds ``threshold``."""
    error = normalized_error(prediction, target)
    exceeded = np.flatnonzero(error > threshold)
    return int(exceeded[0]) if exceeded.size else int(len(error))
