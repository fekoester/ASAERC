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
    Drives the reservoir with ``inp_np`` and correlates what the readout looks at over time.

    Returns:
      C_plain:    correlation of raw values at queries (attention) or sensors (linear)
      C_weighted: correlation of values * scalar query weight
      C_weights:  correlation of the scalar weights alone
    """
    pde_res.reset_state(batch_size=1, device=str(device))
    pde_res.eval()
    readout.eval()

    plain_ts, weighted_ts, weights_ts = [], [], []

    for t in range(inp_np.shape[0]):
        u_t = torch.from_numpy(np.asarray(inp_np[t], dtype=np.float32)).unsqueeze(0).to(device)
        meas_t = pde_res(u_t)  # [1, S]
        field_t = pde_res.state.unsqueeze(0)  # [1, Nx, Ny], the field the readout sees

        _out, q_xy, q_w = readout(meas_t, field_t, return_queries=True)

        if q_xy is None:
            # Linear readout: sensor values, weighted by mean |weight| per sensor.
            p_vals = meas_t.squeeze(0)
            w_t = readout.linear.weight.abs().mean(0)
        else:
            # Attention readout: sample the (normalized) field at the query positions.
            w_t = q_w.squeeze(0).mean(-1)  # [Q]
            mu = field_t.mean(dim=(1, 2), keepdim=True)
            sd = field_t.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            field = ((field_t - mu) / sd).unsqueeze(1)
            grid = (q_xy * 2 - 1).unsqueeze(2)
            p_vals = F.grid_sample(field, grid, mode="bilinear", align_corners=True).reshape(-1)

        plain_ts.append(p_vals.cpu().numpy())
        weighted_ts.append((p_vals * w_t).cpu().numpy())
        weights_ts.append(w_t.cpu().numpy())

    plain_ts = np.vstack(plain_ts)
    weighted_ts = np.vstack(weighted_ts)
    weights_ts = np.vstack(weights_ts)
    return _corrcoef(plain_ts), _corrcoef(weighted_ts), _corrcoef(weights_ts)
