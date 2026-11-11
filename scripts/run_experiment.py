from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from asaerc.diagnostics.corr import compute_correlation_mats
from asaerc.experiment import evaluate_readout, forecast_vpt, simulate_dataset, train_readout
from asaerc.utils.seed import seed_all


@dataclass
class Config:
    processed_dir: str = "processed_data"
    input_dim: int = 4
    output_dim: int = 4
    train_frac: float = 0.8

    # PDE
    Nx: int = 100
    Ny: int = 100
    steps_per_input: int = 25
    n_injection_points: int = 500
    n_measurement_points: int = 128
    use_injection_as_sensors: bool = False
    mode: str = "diffusion"
    boundary: str = "dirichlet"
    c: float = 0.0
    D: float = 9.7e-5
    gamma: float = 0.0
    forcing_type: str = "constant"
    dx: float = 0.0
    dy: float = 0.0
    dt: float = 0.0

    # Readout
    hidden_dim: int = 128
    n_queries: int = 128
    static_positions: bool = False
    use_linear_model: bool = False

    # Train
    lr: float = 1e-2
    final_lr: float = 1e-4
    epochs: int = 250
    batch_size: int = 512
    repulsion_coef: float = 0.0

    # Run
    seed: int = 0
    save_dir: str = "runs"
    save_corr: bool = True
    corr_steps: int = 500  # input steps used for the correlation diagnostics

    # "next": predict y_{t+1}; "delta": predict y_{t+1} - x_t (much harder to game by copying x_t)
    target_mode: str = "next"

    # Closed-loop forecast (0 disables): warm up on the start of each test segment, then predict.
    forecast_steps: int = 0
    forecast_warmup: int = 200


def load_config(path: str) -> Config:
    with open(path) as f:
        d = yaml.safe_load(f)
    return Config(**d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_all(cfg.seed)

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )

    run_dir = Path(cfg.save_dir) / f"run_seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # simulate every file once (seeded reservoir, equal-length files batched)
    pde_res, splits = simulate_dataset(cfg, device)
    if not splits:
        raise RuntimeError("No training data produced. Check processed_dir and data shapes.")

    kind = "linear" if cfg.use_linear_model else ("static" if cfg.static_positions else "adaptive")
    readout, data_hist, repel_hist = train_readout(
        cfg, splits, kind, device, target_mode=cfg.target_mode, return_history=True
    )

    per_file = evaluate_readout(readout, splits, device, part="train")
    per_file_test = evaluate_readout(readout, splits, device, part="test")
    n_train = sum(len(s.train[2]) for s in splits)
    n_test = sum(len(s.test[2]) for s in splits)
    out = {
        "readout": kind,
        "target_mode": cfg.target_mode,
        "train_mse": sum(v["mse"] * len(s.train[2]) for s, v in zip(splits, per_file.values()))
        / n_train,
        "test_mse": (
            sum(
                per_file_test[s.name]["mse"] * len(s.test[2])
                for s in splits
                if s.name in per_file_test
            )
            / n_test
            if n_test
            else float("nan")
        ),
        "data_loss": data_hist,
        "repel_loss": repel_hist,
        "per_file_train_mse": {k: v["mse"] for k, v in per_file.items()},
        "per_file_test_mse": {k: v["mse"] for k, v in per_file_test.items()},
        "per_file_persistence_mse": {k: v["persistence_mse"] for k, v in per_file_test.items()},
    }

    if cfg.forecast_steps > 0:
        out["closed_loop_vpt_steps"] = forecast_vpt(
            pde_res, readout, splits, device, cfg.forecast_warmup, cfg.forecast_steps
        )

    # debug bundles for plotting (first n training points per file)
    debug_n = 1000
    for s in splits:
        n = min(debug_n, s.train[0].shape[0])
        if n <= 0:
            continue
        torch.save(
            {
                "fname": s.name,
                "meas": s.train[0][:n],
                "field": s.train[1][:n],
                "targ": s.train[2][:n],
            },
            run_dir / f"debug_{s.name}.pt",
        )

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    torch.save(readout.state_dict(), run_dir / "readout.pt")
    torch.save(pde_res.state_dict(), run_dir / "pde.pt")

    # correlation diagnostics of what the readout looks at (first file, first corr_steps inputs)
    if cfg.save_corr:
        c_plain, c_weighted, c_weights = compute_correlation_mats(
            pde_res, readout, splits[0].inputs[: cfg.corr_steps], device
        )
        np.savez(
            run_dir / "correlations.npz", plain=c_plain, weighted=c_weighted, weights=c_weights
        )

    print(f"train MSE {out['train_mse']:.5f} | test MSE {out['test_mse']:.5f}")
    print("Done. Saved to:", run_dir)


if __name__ == "__main__":
    main()
