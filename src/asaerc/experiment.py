"""Reusable experiment steps: simulate once, then train/evaluate several readouts on the same data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from asaerc.data import load_data_files_whole
from asaerc.forecast import closed_loop_forecast, valid_prediction_time
from asaerc.readouts import AttentionReadout2DPDE
from asaerc.reservoir import PDE2DReservoir
from asaerc.simulation import simulate_pde_for_files
from asaerc.train import PDETrainer


@dataclass
class FileSplit:
    name: str
    inputs: np.ndarray  # [T, input_dim]
    cutoff: int
    train: tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # meas, field, target
    test: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def target_offset(inputs: torch.Tensor, output_dim: int) -> torch.Tensor:
    """x_t padded/truncated to output_dim: added back to delta predictions."""
    out = torch.zeros(inputs.shape[0], output_dim)
    k = min(inputs.shape[1], output_dim)
    out[:, :k] = inputs[:, :k]
    return out


def simulate_dataset(cfg, device) -> tuple[PDE2DReservoir, list[FileSplit]]:
    """Builds the reservoir (seeded) and simulates every processed file once."""
    file_list = load_data_files_whole(cfg.processed_dir, cfg.input_dim, cfg.output_dim)
    pde_res = PDE2DReservoir(
        input_size=cfg.input_dim,
        n_injection_points=cfg.n_injection_points,
        n_measurement_points=cfg.n_measurement_points,
        use_injection_as_sensors=cfg.use_injection_as_sensors,
        Nx=cfg.Nx,
        Ny=cfg.Ny,
        mode=cfg.mode,
        c=cfg.c,
        D=cfg.D,
        gamma=cfg.gamma,
        boundary=cfg.boundary,
        steps_per_input=cfg.steps_per_input,
        dx=cfg.dx,
        dy=cfg.dy,
        dt=cfg.dt,
        device=str(device),
        forcing_type=cfg.forcing_type,
        seed=cfg.seed,
    )
    simulated = simulate_pde_for_files(pde_res, [(i, t) for _n, i, t in file_list], device)
    splits = []
    for (name, inputs, _targets), (meas, field, targ) in zip(file_list, simulated):
        cutoff = int(cfg.train_frac * meas.shape[0])
        splits.append(
            FileSplit(
                name=name.replace("_data_std.npy", ""),
                inputs=inputs,
                cutoff=cutoff,
                train=(meas[:cutoff], field[:cutoff], targ[:cutoff]),
                test=(meas[cutoff:], field[cutoff:], targ[cutoff:]),
            )
        )
    return pde_res, splits


def _training_target(
    split: FileSplit, part: str, output_dim: int, target_mode: str
) -> torch.Tensor:
    meas, _field, targ = getattr(split, part)
    if target_mode == "delta":
        start = 0 if part == "train" else split.cutoff
        x = torch.from_numpy(split.inputs[start : start + len(targ)]).float()
        return targ - target_offset(x, output_dim)
    return targ


def train_readout(
    cfg,
    splits: list[FileSplit],
    kind: str,
    device,
    target_mode: str = "next",
    return_history: bool = False,
):
    """kind: 'linear' | 'static' (fixed query positions) | 'adaptive' (learned query positions)."""
    readout = AttentionReadout2DPDE(
        num_sensors=cfg.n_measurement_points,
        output_dim=cfg.output_dim,
        n_queries=cfg.n_queries,
        hidden_dim=cfg.hidden_dim,
        static_positions=kind == "static",
        use_linear_model=kind == "linear",
    ).to(device)
    meas = torch.cat([s.train[0] for s in splits])
    field = torch.cat([s.train[1] for s in splits])
    targ = torch.cat([_training_target(s, "train", cfg.output_dim, target_mode) for s in splits])
    loader = DataLoader(
        TensorDataset(meas, field, targ), batch_size=min(cfg.batch_size, 64), shuffle=True
    )
    trainer = PDETrainer(readout, None, lr=cfg.lr, final_lr=cfg.final_lr, device=device)
    history = trainer.train_loop(
        loader, n_epochs=cfg.epochs, repulsion_coef=cfg.repulsion_coef, verbose=return_history
    )
    readout.target_mode = target_mode
    if return_history:
        return readout, *history
    return readout


@torch.no_grad()
def evaluate_readout(
    readout, splits: list[FileSplit], device, part: str = "test", batch_size: int = 512
) -> dict[str, dict[str, float]]:
    """Per-file MSE of the readout and of the persistence baseline (y_{t+1} = x_t)."""
    readout.eval()
    results = {}
    for s in splits:
        meas, field, targ = getattr(s, part)
        if len(targ) == 0:
            continue
        pred = torch.cat(
            [
                readout(meas[i : i + batch_size].to(device), field[i : i + batch_size].to(device))
                .float()
                .cpu()
                for i in range(0, len(targ), batch_size)
            ]
        )
        start = 0 if part == "train" else s.cutoff
        x = target_offset(
            torch.from_numpy(s.inputs[start : start + len(targ)]).float(), targ.shape[1]
        )
        if getattr(readout, "target_mode", "next") == "delta":
            pred = pred + x
        results[s.name] = {
            "mse": float(torch.mean((pred - targ) ** 2)),
            "persistence_mse": float(torch.mean((x - targ) ** 2)),
        }
    return results


def forecast_vpt(pde_res, readout, splits, device, warmup: int, steps: int) -> dict[str, int]:
    """Closed-loop valid prediction time (steps) at the start of each test segment."""
    vpt = {}
    delta = getattr(readout, "target_mode", "next") == "delta"
    for s in splits:
        start = s.cutoff
        if start + warmup + steps >= len(s.inputs):
            continue
        warm = s.inputs[start : start + warmup + 1]
        pred = closed_loop_forecast(pde_res, readout, warm, steps, device, delta=delta)
        truth = s.inputs[start + warmup + 1 : start + warmup + 1 + steps]
        k = min(pred.shape[1], truth.shape[1])
        vpt[s.name] = valid_prediction_time(pred[:, :k], truth[:, :k])
    return vpt
