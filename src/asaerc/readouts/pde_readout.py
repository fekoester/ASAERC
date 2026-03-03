from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionReadout2DPDE(nn.Module):
    """
    Readout for PDE reservoir.
    Inputs:
      - measurement: [B, num_sensors]
      - pde_reservoir.state: [Nx, Ny] (sampled via grid_sample)

    Modes:
      - use_linear_model=True: linear(measurement)->output
      - else: attention queries (static or learned) over PDE field
    """

    def __init__(
        self,
        num_sensors: int,
        output_dim: int,
        n_queries: int = 10,
        hidden_dim: int = 64,
        static_positions: bool = False,
        use_linear_model: bool = False,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.output_dim = output_dim
        self.n_queries = n_queries
        self.static_positions = static_positions
        self.use_linear_model = use_linear_model

        if use_linear_model:
            self.linear = nn.Linear(num_sensors, output_dim)
        else:
            if static_positions:
                self.register_buffer("fixed_xy", torch.rand(n_queries, 2))
                self.net = nn.Sequential(
                    nn.Linear(num_sensors, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, n_queries * output_dim),
                )
            else:
                self.net = nn.Sequential(
                    nn.Linear(num_sensors, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, n_queries * (2 + output_dim)),
                )

    def forward(
        self, measurement: torch.Tensor, field_u: torch.Tensor, return_queries: bool = False
    ):
        """
        measurement: [B, num_sensors]   fixed sensors measured at u_{t+T}
        field_u:     [B, Nx, Ny]        the PDE field u_{t+T} for each sample
        """
        B = measurement.size(0)

        # Linear readout ignores field_u but we keep signature consistent
        if self.use_linear_model:
            out = self.linear(measurement)
            if return_queries:
                return out, None, None
            return out

        raw = self.net(measurement)

        if self.static_positions:
            raw_weights = raw.view(B, self.n_queries, self.output_dim)
            query_xy = self.fixed_xy.unsqueeze(0).expand(B, -1, -1)
        else:
            raw = raw.view(B, self.n_queries, 2 + self.output_dim)
            query_xy = torch.sigmoid(raw[:, :, :2])
            raw_weights = raw[:, :, 2:]

        # field_u: [B,Nx,Ny]
        mu = field_u.mean(dim=(1, 2), keepdim=True)
        sd = field_u.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        field_u_norm = (field_u - mu) / sd
        field_4d = field_u_norm.unsqueeze(1).float()

        # query_xy in [0,1] -> [-1,1] for grid_sample
        grid = (query_xy * 2.0 - 1.0).unsqueeze(2)  # [B,Q,1,2]
        sampled = F.grid_sample(field_4d, grid, mode="bilinear", align_corners=True)
        pde_vals = sampled.squeeze(1).squeeze(-1)  # [B,Q]

        out = (pde_vals.unsqueeze(-1) * raw_weights).sum(dim=1)  # [B, out_dim]

        if return_queries:
            return out, query_xy, raw_weights
        return out
