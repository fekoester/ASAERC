from __future__ import annotations


import numpy as np
import torch
import torch.nn.functional as F


def _corrcoef(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(0, keepdims=True)
    x /= x.std(0, keepdims=True) + 1e-12
    return (x.T @ x) / (x.shape[0] - 1)


@torch.no_grad()
def compute_correlation_mats(pde_res, readout, inp_np: np.ndarray, device: torch.device):
    """
    Returns:
      C_plain:    correlation of raw values at queries/sensors
      C_weighted: correlation of values * scalar(query-weight)
      C_weights:  correlation of scalar weights alone
    """
    pde_res.reset_state(batch_size=1, device=str(device))
    pde_res.eval()
    readout.eval()

    plain_ts, weighted_ts, weights_ts = [], [], []

    # Determine mode by probing return_queries on first real step (no burn).
    for t in range(inp_np.shape[0]):
        u_t = torch.from_numpy(inp_np[t]).float().unsqueeze(0).to(device)
        meas_t = pde_res(u_t)

        out, q_xy, q_w = readout(meas_t, pde_res, return_queries=True)

        if q_xy is None:
            # linear readout path: use sensors and derive per-sensor scalar weights from linear layer
            p_vals = meas_t.squeeze(0)  # [S]
            w_query = readout.linear.weight.abs().mean(0)  # [S]
            w_t = w_query
        else:
            # attention: sample PDE field at queries
            w_t = q_w.squeeze(0).mean(-1)  # [Q] scalar per query
            field = pde_res.state.unsqueeze(0).unsqueeze(0)
            grid = (q_xy * 2 - 1).unsqueeze(2)
            sampled = F.grid_sample(
                field.expand(1, 1, field.shape[-2], field.shape[-1]),
                grid,
                mode="bilinear",
                align_corners=True,
            )
            p_vals = sampled.squeeze()  # [Q]

        plain_ts.append(p_vals.detach().cpu().numpy())
        weighted_ts.append((p_vals * w_t).detach().cpu().numpy())
        weights_ts.append(w_t.detach().cpu().numpy())

    plain_ts = np.vstack(plain_ts)
    weighted_ts = np.vstack(weighted_ts)
    weights_ts = np.vstack(weights_ts)

    return _corrcoef(plain_ts), _corrcoef(weighted_ts), _corrcoef(weights_ts)
