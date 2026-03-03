from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from asaerc import (
    AttentionReadout2DPDE,
    build_pde_reservoir_compiled,
    load_data_files_whole,
    simulate_pde_for_file,
)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_loss(metrics: dict, out_png: Path) -> None:
    loss = metrics.get("data_loss", [])
    repel = metrics.get("repel_loss", [])
    plt.figure()
    if loss:
        plt.plot(loss, label="data_loss")
    if repel and any(x != 0.0 for x in repel):
        plt.plot(repel, label="repel_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_pred_vs_target(
    run_dir: Path,
    cfg: dict,
    device: torch.device,
    out_png: Path,
    n_steps: int = 1000,
    file_name=None,
) -> None:
    # rebuild + load models (readout matters; PDE object only needed for metadata like num_sensors)
    pde = build_pde_reservoir_compiled(
        input_size=cfg["input_dim"],
        n_injection_points=cfg["n_injection_points"],
        n_measurement_points=cfg["n_measurement_points"],
        use_injection_as_sensors=cfg["use_injection_as_sensors"],
        Nx=cfg["Nx"],
        Ny=cfg["Ny"],
        mode=cfg["mode"],
        c=cfg["c"],
        D=cfg["D"],
        gamma=cfg["gamma"],
        boundary=cfg["boundary"],
        steps_per_input=cfg["steps_per_input"],
        dx=cfg["dx"],
        dy=cfg["dy"],
        dt=cfg["dt"],
        device=str(device),
        forcing_type=cfg["forcing_type"],
    )
    pde.load_state_dict(torch.load(run_dir / "pde.pt", map_location=device))

    readout = AttentionReadout2DPDE(
        num_sensors=pde.num_sensors,
        output_dim=cfg["output_dim"],
        n_queries=cfg["n_queries"],
        hidden_dim=cfg["hidden_dim"],
        static_positions=cfg["static_positions"],
        use_linear_model=cfg["use_linear_model"],
    ).to(device)
    readout.load_state_dict(torch.load(run_dir / "readout.pt", map_location=device))
    readout.eval()

    # --- Load cached debug bundle produced during training (guaranteed aligned) ---
    if file_name is None:
        debug_files = sorted(run_dir.glob("debug_*.pt"))
        if not debug_files:
            raise FileNotFoundError(
                f"No debug bundles found in {run_dir}. "
                f"Re-run run_experiment.py after enabling saving debug_<fname>.pt bundles."
            )
        bundle_path = debug_files[0]
        bundle = torch.load(bundle_path, map_location="cpu")
        fname = bundle["fname"]
    else:
        fname = file_name
        bundle_path = run_dir / f"debug_{fname}.pt"
        if not bundle_path.exists():
            available = sorted(
                p.name.replace("debug_", "").replace(".pt", "") for p in run_dir.glob("debug_*.pt")
            )
            raise FileNotFoundError(
                f"Missing debug bundle: {bundle_path}\nAvailable bundles: {available}"
            )
        bundle = torch.load(bundle_path, map_location="cpu")

    meas = bundle["meas"]  # [T, S]
    field = bundle["field"]  # [T, Nx, Ny]
    targ = bundle["targ"]  # [T, out_dim]

    n = min(n_steps, targ.shape[0])
    meas = meas[:n]
    field = field[:n]
    targ = targ[:n]

    with torch.no_grad():
        pred = readout(meas.to(device), field.to(device)).cpu()

    out_dim = cfg["output_dim"]
    kmax = min(out_dim, 4)

    plt.figure(figsize=(12, 2.8 * kmax))
    for k in range(kmax):
        ax = plt.subplot(kmax, 1, k + 1)
        ax.plot(targ[:, k].numpy(), label="target", linewidth=1.0)
        ax.plot(pred[:, k].numpy(), label="pred", linewidth=1.0)
        ax.set_title(f"{fname}  dim={k}")
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_corr_heatmaps(run_dir: Path, plots_dir: Path) -> None:
    corr_dir = run_dir / "corr"
    if not corr_dir.exists():
        return

    # pick one file (first *_plain.npy)
    plain_files = sorted(corr_dir.glob("*_plain.npy"))
    if not plain_files:
        return

    plain_path = plain_files[0]
    base = plain_path.name.replace("_plain.npy", "")

    C_plain = np.load(plain_path)
    C_weighted = np.load(corr_dir / f"{base}_weighted.npy")
    C_w = np.load(corr_dir / f"{base}_weights.npy")

    def _save_heat(C: np.ndarray, title: str, out_png: Path) -> None:
        plt.figure(figsize=(6, 5))
        plt.imshow(C, aspect="auto", interpolation="nearest")
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.close()

    _save_heat(C_plain, f"{base} corr: plain", plots_dir / f"{base}_corr_plain.png")
    _save_heat(C_weighted, f"{base} corr: weighted", plots_dir / f"{base}_corr_weighted.png")
    _save_heat(C_w, f"{base} corr: weights", plots_dir / f"{base}_corr_weights.png")


def plot_query_positions(run_dir: Path, cfg: dict, device: torch.device, out_png: Path) -> None:
    if cfg.get("use_linear_model", False):
        return

    pde = build_pde_reservoir_compiled(
        input_size=cfg["input_dim"],
        n_injection_points=cfg["n_injection_points"],
        n_measurement_points=cfg["n_measurement_points"],
        use_injection_as_sensors=cfg["use_injection_as_sensors"],
        Nx=cfg["Nx"],
        Ny=cfg["Ny"],
        mode=cfg["mode"],
        c=cfg["c"],
        D=cfg["D"],
        gamma=cfg["gamma"],
        boundary=cfg["boundary"],
        steps_per_input=cfg["steps_per_input"],
        dx=cfg["dx"],
        dy=cfg["dy"],
        dt=cfg["dt"],
        device=str(device),
        forcing_type=cfg["forcing_type"],
    )
    pde.load_state_dict(torch.load(run_dir / "pde.pt", map_location=device))

    readout = AttentionReadout2DPDE(
        num_sensors=pde.num_sensors,
        output_dim=cfg["output_dim"],
        n_queries=cfg["n_queries"],
        hidden_dim=cfg["hidden_dim"],
        static_positions=cfg["static_positions"],
        use_linear_model=cfg["use_linear_model"],
    ).to(device)
    readout.load_state_dict(torch.load(run_dir / "readout.pt", map_location=device))
    readout.eval()

    files = load_data_files_whole(cfg["processed_dir"], cfg["input_dim"], cfg["output_dim"])
    _, x, y = files[0]

    # simulate one step to get a valid (meas, field)
    meas, field, _ = simulate_pde_for_file(pde, x[:1], y[:1], device=device)
    meas0 = meas.to(device)  # [1,S]
    field0 = field.to(device)  # [1,Nx,Ny]

    with torch.no_grad():
        _, q_xy, _ = readout(meas0, field0, return_queries=True)

    if q_xy is None:
        return

    q = q_xy.squeeze(0).cpu().numpy()
    sensors = pde.sensor_positions.detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(sensors[:, 0], sensors[:, 1], s=10, alpha=0.6, label="sensors")
    plt.scatter(q[:, 0], q[:, 1], s=20, alpha=0.9, label="queries")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Sensor positions and attention queries")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n_steps", type=int, default=1000)
    ap.add_argument(
        "--file", type=str, default=None, help="Specific filename (e.g. lorenz_data_std.npy)"
    )
    ap.add_argument("--all", action="store_true", help="Plot predictions for all processed files")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = json.load(open(run_dir / "config.json"))
    metrics = json.load(open(run_dir / "metrics.json"))

    device = torch.device(
        args.device if (torch.cuda.is_available() and args.device.startswith("cuda")) else "cpu"
    )

    plots_dir = run_dir / "plots"
    ensure_dir(plots_dir)

    plot_loss(metrics, plots_dir / "loss.png")
    if args.all:
        files = load_data_files_whole(cfg["processed_dir"], cfg["input_dim"], cfg["output_dim"])
        for fname, _, _ in files:
            out = plots_dir / f"pred_vs_target__{fname.replace('.npy', '')}.png"
            plot_pred_vs_target(run_dir, cfg, device, out, n_steps=args.n_steps, file_name=fname)
    else:
        plot_pred_vs_target(
            run_dir,
            cfg,
            device,
            plots_dir / "pred_vs_target.png",
            n_steps=args.n_steps,
            file_name=args.file,
        )
    plot_query_positions(run_dir, cfg, device, plots_dir / "query_positions.png")
    plot_corr_heatmaps(run_dir, plots_dir)

    print("Saved plots to:", plots_dir)


if __name__ == "__main__":
    main()
