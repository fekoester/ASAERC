"""
Compare linear, static-query (AERC) and adaptive-sensing readouts on the same PDE reservoir.

For every seed the reservoir is built and all series are simulated once; each readout is then
trained on identical data. Writes results.csv (one row per seed/readout/target_mode/system),
summary.md (mean ± std over seeds, relative to the persistence baseline) and comparison.png.

    python scripts/compare_readouts.py --config configs/example.yaml --seeds 0,1,2 \\
        --target-modes next,delta --epochs 30 --out results/readout_comparison
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from run_experiment import load_config  # noqa: E402

from asaerc.experiment import (  # noqa: E402
    evaluate_readout,
    forecast_vpt,
    simulate_dataset,
    train_readout,
)
from asaerc.utils.seed import seed_all  # noqa: E402

KINDS = ["linear", "static", "adaptive"]
FIELDS = ["seed", "target_mode", "readout", "system", "test_mse", "persistence_mse", "vpt_steps"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--readouts", default=",".join(KINDS))
    ap.add_argument("--target-modes", default="next,delta")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--forecast-steps", type=int, default=100)
    ap.add_argument("--forecast-warmup", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/readout_comparison")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg = replace(cfg, epochs=args.epochs)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        seed_all(seed)
        cfg_seed = replace(cfg, seed=seed)
        pde_res, splits = simulate_dataset(cfg_seed, device)
        for target_mode in args.target_modes.split(","):
            for kind in args.readouts.split(","):
                seed_all(seed)
                readout = train_readout(cfg_seed, splits, kind, device, target_mode=target_mode)
                scores = evaluate_readout(readout, splits, device)
                vpt = forecast_vpt(
                    pde_res, readout, splits, device, args.forecast_warmup, args.forecast_steps
                )
                for system, s in scores.items():
                    rows.append(
                        dict(
                            seed=seed,
                            target_mode=target_mode,
                            readout=kind,
                            system=system,
                            test_mse=s["mse"],
                            persistence_mse=s["persistence_mse"],
                            vpt_steps=vpt.get(system, ""),
                        )
                    )
                mean_mse = np.mean([s["mse"] for s in scores.values()])
                print(
                    f"seed={seed} mode={target_mode:5s} readout={kind:8s} mean test MSE={mean_mse:.5f}"
                )

    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summarize(rows, out_dir)


def _fmt(values: list[float], digits: int = 4) -> str:
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def summarize(rows: list[dict], out_dir: Path) -> None:
    modes = sorted({r["target_mode"] for r in rows})
    kinds = [k for k in KINDS if any(r["readout"] == k for r in rows)]
    systems = sorted({r["system"] for r in rows})

    def per_seed(mode, kind, key, system=None):
        values = {}
        for r in rows:
            if (
                r["target_mode"] == mode
                and r["readout"] == kind
                and (system is None or r["system"] == system)
            ):
                if r[key] != "":
                    values.setdefault(r["seed"], []).append(float(r[key]))
        return [float(np.mean(v)) for v in values.values()]

    lines = [
        "## Mean over all systems",
        "",
        "| target | readout | test MSE | persistence MSE | VPT (steps) |",
    ]
    lines.append("|---|---|---:|---:|---:|")
    for mode in modes:
        for kind in kinds:
            lines.append(
                f"| {mode} | {kind} | {_fmt(per_seed(mode, kind, 'test_mse'))} | "
                f"{_fmt(per_seed(mode, kind, 'persistence_mse'))} | "
                f"{_fmt(per_seed(mode, kind, 'vpt_steps'), 1)} |"
            )
    lines += [
        "",
        "## Test MSE per system",
        "",
        "| system | target | " + " | ".join(kinds) + " | persistence |",
    ]
    lines.append("|---|---|" + "---:|" * (len(kinds) + 1))
    for system in systems:
        for mode in modes:
            cells = [_fmt(per_seed(mode, k, "test_mse", system)) for k in kinds]
            persistence = _fmt(per_seed(mode, kinds[0], "persistence_mse", system))
            lines.append(f"| {system} | {mode} | " + " | ".join(cells) + f" | {persistence} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4.5), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        x = np.arange(len(systems))
        width = 0.8 / (len(kinds) + 1)
        for i, kind in enumerate(kinds):
            means = [np.mean(per_seed(mode, kind, "test_mse", s)) for s in systems]
            stds = [np.std(per_seed(mode, kind, "test_mse", s)) for s in systems]
            ax.bar(x + i * width, means, width, yerr=stds, label=kind)
        persistence = [np.mean(per_seed(mode, kinds[0], "persistence_mse", s)) for s in systems]
        ax.bar(x + len(kinds) * width, persistence, width, label="persistence", color="0.6")
        ax.set_yscale("log")
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels(systems, rotation=35, ha="right")
        ax.set_ylabel("one-step test MSE (standardized)")
        ax.set_title(f"target_mode = {mode}")
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=150)


if __name__ == "__main__":
    main()
