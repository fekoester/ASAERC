"""
Animate the PDE reservoir field for a trained run, with the readout's query positions on top.

    python scripts/animate_field.py --run_dir runs/run_seed0 --system lorenz --steps 150 \\
        --out docs/media/field.gif

Left: the reservoir field u(x, y, t) driven by the chosen series. Dots: sensors (white) and, for the
adaptive readout, the learned query positions (coloured by their output weight), which move with
the state. Right: target vs. one-step prediction of the first output dimension.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from asaerc import AttentionReadout2DPDE, PDE2DReservoir  # noqa: E402
from asaerc.data import load_data_files_whole  # noqa: E402


def load_run(run_dir: Path, device: str):
    cfg = json.loads((run_dir / "config.json").read_text())
    res = PDE2DReservoir(
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
        device=device,
        forcing_type=cfg["forcing_type"],
        seed=cfg.get("seed"),
    )
    res.load_state_dict(torch.load(run_dir / "pde.pt", map_location=device))
    readout = AttentionReadout2DPDE(
        num_sensors=res.num_sensors,
        output_dim=cfg["output_dim"],
        n_queries=cfg["n_queries"],
        hidden_dim=cfg["hidden_dim"],
        static_positions=cfg["static_positions"],
        use_linear_model=cfg["use_linear_model"],
    )
    readout.load_state_dict(torch.load(run_dir / "readout.pt", map_location=device))
    return cfg, res.eval(), readout.eval()


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--system", default="lorenz", help="file prefix, e.g. lorenz, roessler")
    ap.add_argument(
        "--start",
        type=int,
        default=None,
        help="first warm-up step (default: test segment start - warmup)",
    )
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", default="field.gif")
    ap.add_argument("--dpi", type=int, default=80)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg, res, readout = load_run(Path(args.run_dir), args.device)
    files = {
        name.replace("_data_std.npy", ""): (inp, tg)
        for name, inp, tg in load_data_files_whole(
            cfg["processed_dir"], cfg["input_dim"], cfg["output_dim"]
        )
    }
    inputs, targets = files[args.system]
    delta = cfg.get("target_mode", "next") == "delta"
    if args.start is None:  # animate the held-out test segment
        args.start = max(0, int(cfg.get("train_frac", 0.8) * len(inputs)) - args.warmup)

    res.reset_state(batch_size=1, device=args.device)
    for u in inputs[args.start : args.start + args.warmup]:
        res(torch.from_numpy(np.asarray(u, np.float32))[None])

    frames = []
    first = args.start + args.warmup
    for t in range(first, first + args.steps):
        meas = res(torch.from_numpy(np.asarray(inputs[t], np.float32))[None])
        field = res.state.clone()
        pred, q_xy, q_w = readout(meas, field[None], return_queries=True)
        pred = pred[0].numpy()
        if delta:
            pred[: inputs.shape[1]] += inputs[t][: pred.shape[0]]
        frames.append(
            dict(
                field=field.numpy(),
                queries=None if q_xy is None else q_xy[0].numpy(),
                weights=None if q_w is None else q_w[0, :, 0].numpy(),
                pred=float(pred[0]),
                target=float(targets[t][0]),
            )
        )

    sensors = res.sensor_positions.numpy()
    vmax = np.percentile(np.abs(np.stack([f["field"] for f in frames])), 99)
    fig, (ax_field, ax_ts) = plt.subplots(
        1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1.25]}
    )
    image = ax_field.imshow(
        frames[0]["field"].T,
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax_field.scatter(
        sensors[:, 0],
        sensors[:, 1],
        s=6,
        c="white",
        edgecolors="k",
        linewidths=0.3,
        label="sensors",
    )
    query_plot = ax_field.scatter(
        [], [], s=22, c=[], cmap="viridis", edgecolors="k", linewidths=0.4, label="adaptive queries"
    )
    ax_field.set_xticks([])
    ax_field.set_yticks([])
    ax_field.set_title(f"PDE reservoir ({cfg['mode']}), input: {args.system}")
    ax_field.legend(loc="upper right", fontsize=7, framealpha=0.8)

    steps = np.arange(len(frames))
    ax_ts.plot(steps, [f["target"] for f in frames], color="0.3", lw=1.5, label="target")
    (pred_line,) = ax_ts.plot([], [], color="tab:red", lw=1.5, label="one-step prediction")
    marker = ax_ts.axvline(0, color="0.6", lw=0.8)
    ax_ts.set_xlabel("input step (held-out test segment)")
    ax_ts.set_ylabel(f"{args.system} x-component (standardized)")
    ax_ts.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    def update(i):
        frame = frames[i]
        image.set_data(frame["field"].T)
        if frame["queries"] is not None:
            query_plot.set_offsets(frame["queries"])
            query_plot.set_array(frame["weights"])
        pred_line.set_data(steps[: i + 1], [f["pred"] for f in frames[: i + 1]])
        marker.set_xdata([i, i])
        return image, query_plot, pred_line, marker

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    animation.save(args.out, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print("saved", args.out)


if __name__ == "__main__":
    main()
