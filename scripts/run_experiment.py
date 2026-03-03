from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from asaerc import (
    AttentionReadout2DPDE,
    PDETrainer,
    build_pde_reservoir_compiled,
    load_data_files_whole,
    simulate_pde_for_file,
)
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


def load_config(path: str) -> Config:
    with open(path, "r") as f:
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

    # load data
    file_list = load_data_files_whole(cfg.processed_dir, cfg.input_dim, cfg.output_dim)

    # build reservoir
    pde_res = build_pde_reservoir_compiled(
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
    )

    # simulate each file, create per-file splits
    filewise = {}
    train_meas, train_field, train_targ = [], [], []
    for fname, arr_in, arr_tg in file_list:
        meas_cpu, field_cpu, targ_cpu = simulate_pde_for_file(
            pde_res, arr_in, arr_tg, device=device
        )
        T = meas_cpu.size(0)
        if T < 2:
            continue
        cutoff = int(cfg.train_frac * T)

        filewise[fname] = dict(
            train_meas=meas_cpu[:cutoff],
            train_field=field_cpu[:cutoff],
            train_targ=targ_cpu[:cutoff],
            test_meas=meas_cpu[cutoff:],
            test_field=field_cpu[cutoff:],
            test_targ=targ_cpu[cutoff:],
            inp_np=arr_in,
        )

        train_meas.append(meas_cpu[:cutoff])
        train_field.append(field_cpu[:cutoff])
        train_targ.append(targ_cpu[:cutoff])

    if len(train_meas) == 0:
        raise RuntimeError("No training data produced. Check processed_dir and data shapes.")

    all_train_meas = torch.cat(train_meas, dim=0)
    all_train_field = torch.cat(train_field, dim=0)
    all_train_targ = torch.cat(train_targ, dim=0)

    # readout + trainer
    readout = AttentionReadout2DPDE(
        num_sensors=pde_res.num_sensors,  # source of truth
        output_dim=cfg.output_dim,
        n_queries=cfg.n_queries,
        hidden_dim=cfg.hidden_dim,
        static_positions=cfg.static_positions,
        use_linear_model=cfg.use_linear_model,
    ).to(device)

    # IMPORTANT: dataset includes fields (u_{t+T})
    # NOTE: fields are float16 CPU tensors; move to GPU per batch.
    loader = DataLoader(
        TensorDataset(all_train_meas, all_train_field, all_train_targ),
        batch_size=min(cfg.batch_size, 64),  # fields are big; keep this moderate
        shuffle=True,
        drop_last=False,
    )

    trainer = PDETrainer(readout, pde_res, lr=cfg.lr, final_lr=cfg.final_lr, device=device)
    data_hist, repel_hist = trainer.train_loop(
        loader, n_epochs=cfg.epochs, repulsion_coef=cfg.repulsion_coef
    )

    # metrics
    readout.eval()
    mse = torch.nn.MSELoss()

    # Global train MSE
    with torch.no_grad():
        pred_train = readout(all_train_meas.to(device), all_train_field.to(device)).cpu()
        train_mse = mse(pred_train, all_train_targ).item()

    # Global test MSE
    test_meas, test_field, test_targ = [], [], []
    for d in filewise.values():
        if d["test_meas"].numel():
            test_meas.append(d["test_meas"])
            test_field.append(d["test_field"])
            test_targ.append(d["test_targ"])

    if test_meas:
        all_test_meas = torch.cat(test_meas, dim=0)
        all_test_field = torch.cat(test_field, dim=0)
        all_test_targ = torch.cat(test_targ, dim=0)
        with torch.no_grad():
            pred_test = readout(all_test_meas.to(device), all_test_field.to(device)).cpu()
            test_mse = mse(pred_test, all_test_targ).item()
    else:
        test_mse = float("nan")

    # Build metrics dict early
    out = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "data_loss": data_hist,
        "repel_loss": repel_hist,
    }

    # --- Per-file MSE (sanity check) ---
    per_file = {}
    with torch.no_grad():
        for fname, d in filewise.items():
            tm = d["train_meas"]
            tf = d["train_field"]
            tt = d["train_targ"]

            if tm.numel() == 0:
                per_file[fname] = float("nan")
                continue

            pred = readout(tm.to(device), tf.to(device)).cpu()
            per_file[fname] = mse(pred, tt).item()

    out["per_file_train_mse"] = per_file

    # --- Save debug bundles for plotting (first n points per file) ---
    debug_n = 1000
    for fname, d in filewise.items():
        n = min(debug_n, d["train_meas"].shape[0])
        if n <= 0:
            continue
        torch.save(
            {
                "fname": fname,
                "meas": d["train_meas"][:n],
                "field": d["train_field"][:n],
                "targ": d["train_targ"][:n],
            },
            run_dir / f"debug_{fname}.pt",
        )

    # Finally write metrics.json once, with everything included
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    # save models
    torch.save(readout.state_dict(), run_dir / "readout.pt")
    torch.save(pde_res.state_dict(), run_dir / "pde.pt")

    # optional correlations: DISABLE for now unless you update compute_correlation_mats to accept fields
    # Current diagnostics module likely assumes readout(meas, pde_res). Leave off to avoid silent wrong results.
    if cfg.save_corr:
        print(
            "NOTE: save_corr=True but compute_correlation_mats must be updated for (meas, field_u). Skipping corr."
        )

    print("Done. Saved to:", run_dir)


if __name__ == "__main__":
    main()
