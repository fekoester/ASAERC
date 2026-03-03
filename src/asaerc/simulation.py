from __future__ import annotations


import torch


@torch.no_grad()
def simulate_pde_for_file(pde_res, in_np, tg_np, device, field_dtype=torch.float16):
    """
    Returns:
      meas_cpu : [T, S]      fixed sensors at u_{t+T}
      field_cpu: [T, Nx, Ny] PDE field u_{t+T} per sample
      targ_cpu : [T, out]
    """
    pde_res.eval()

    in_torch = torch.from_numpy(in_np.copy()).float().to(device)
    tg_torch = torch.from_numpy(tg_np.copy()).float().to(device)

    pde_res.reset_state(batch_size=1, device=str(device))

    meas_list = []
    field_list = []

    for t in range(in_torch.size(0)):
        inp_t = in_torch[t].unsqueeze(0)  # [1,input_dim]
        meas_t = pde_res(inp_t)  # evolves to u_{t+T}, returns sensors at u_{t+T}
        meas_list.append(meas_t.cpu())

        # store u_{t+T}
        field_list.append(pde_res.state.detach().to(field_dtype).cpu())

    meas_cpu = torch.cat(meas_list, dim=0)  # [T,S]
    field_cpu = torch.stack(field_list, dim=0)  # [T,Nx,Ny]
    targ_cpu = tg_torch.cpu()  # [T,out]
    return meas_cpu, field_cpu, targ_cpu
