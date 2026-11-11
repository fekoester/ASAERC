from __future__ import annotations

import torch


@torch.no_grad()
def simulate_pde_for_file(pde_res, in_np, tg_np, device, field_dtype=torch.float16):
    """
    Drives the reservoir with one input series.

    Returns:
      meas_cpu : [T, S]      fixed sensors at u_{t+T}
      field_cpu: [T, Nx, Ny] PDE field u_{t+T} per sample
      targ_cpu : [T, out]
    """
    results = simulate_pde_for_files(pde_res, [(in_np, tg_np)], device, field_dtype)
    return results[0]


@torch.no_grad()
def simulate_pde_for_files(pde_res, series, device, field_dtype=torch.float16):
    """
    Drives the reservoir with several independent input series at once.

    ``series`` is a list of (inputs [T_i, input_dim], targets [T_i, out]). Series of equal length are
    simulated as one batch of independent PDE fields, which is much faster than one by one; results
    are identical to simulating each series separately from a zero field.
    """
    pde_res.eval()
    results: list = [None] * len(series)

    by_length: dict[int, list[int]] = {}
    for index, (inputs, _targets) in enumerate(series):
        by_length.setdefault(len(inputs), []).append(index)

    for length, indices in by_length.items():
        inputs = torch.stack(
            [torch.from_numpy(series[i][0].copy()).float() for i in indices], dim=1
        ).to(device)  # [T, B, input_dim]
        batch = len(indices)
        pde_res.reset_state(batch_size=batch, device=str(device))

        meas = []
        fields = []
        for t in range(length):
            meas_t = pde_res(inputs[t])  # [B, S]
            meas.append(meas_t.cpu())
            state = pde_res.state if batch > 1 else pde_res.state.unsqueeze(0)
            fields.append(state.detach().to(field_dtype).cpu())

        meas_all = torch.stack(meas, dim=1)  # [B, T, S]
        field_all = torch.stack(fields, dim=1)  # [B, T, Nx, Ny]
        for b, index in enumerate(indices):
            targets = torch.from_numpy(series[index][1].copy()).float()
            results[index] = (meas_all[b], field_all[b], targets)

    return results
