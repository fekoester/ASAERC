from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch import compile as torch_compile  # torch>=2

    HAS_TORCH_COMPILE = True
except Exception:
    HAS_TORCH_COMPILE = False


@torch.no_grad()
def _laplacian_2d(U: torch.Tensor, conv2d: nn.Conv2d, pad_mode: str = "circular") -> torch.Tensor:
    """Compute Laplacian via 2D conv. U: [Nx, Ny] -> [Nx, Ny]."""
    U_4d = U.unsqueeze(0).unsqueeze(0)  # [1,1,Nx,Ny]
    lap = F.pad(U_4d, (1, 1, 1, 1), mode=pad_mode)
    lap = conv2d(lap)
    return lap.squeeze(0).squeeze(0)


def _evolve_pde(
    U: torch.Tensor,
    V: Optional[torch.Tensor],
    forcing_sequence: List[torch.Tensor],
    steps: int,
    mode: str,
    pad_mode: str,
    dt: float,
    D: float,
    alpha: float,
    c: float,
    gamma: float,
    laplacian_conv: nn.Conv2d,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Micro-step loop. Kept as a single function to benefit from torch.compile."""
    for i in range(steps):
        f_i = forcing_sequence[i]
        if mode == "diffusion":
            lapU = _laplacian_2d(U, laplacian_conv, pad_mode)
            U = U + dt * (D * lapU + f_i)

        elif mode == "diffusion_leaky":
            lapU = _laplacian_2d(U, laplacian_conv, pad_mode)
            U = alpha * U + dt * (D * lapU + f_i)

        elif mode == "wave":
            if V is None:
                V = torch.zeros_like(U)
            lapU = _laplacian_2d(U, laplacian_conv, pad_mode)
            accel = c**2 * lapU - gamma * V + f_i
            V = V + dt * accel
            U = U + dt * V
        else:
            raise ValueError(f"Unknown PDE mode: {mode}")

    return U, V


class PDE2DReservoir(nn.Module):
    """
    PDE reservoir (batch_size=1) with:
      - forcing_mask built from random injection points (numpy RNG)
      - partial sensor measurements by fancy indexing
    """

    def __init__(
        self,
        input_size: int,
        n_injection_points: int,
        n_measurement_points: int,
        use_injection_as_sensors: bool = True,
        Nx: int = 50,
        Ny: int = 50,
        D: float = 0.01,
        alpha: float = 1.0,
        c: float = 0.1,
        gamma: float = 0.01,
        dx: float = 0.02,
        dy: float = 0.02,
        dt: float = 0.001,
        steps_per_input: int = 10,
        boundary: str = "periodic",
        mode: str = "diffusion",
        device: str = "cuda",
        forcing_type: str = "periodic",
        use_optimal_step_size: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_injection_points = n_injection_points
        self.n_measurement_points = n_measurement_points
        self.use_injection_as_sensors = use_injection_as_sensors
        self.forcing_type = forcing_type

        self.Nx = Nx
        self.Ny = Ny
        self.D = D
        self.alpha = alpha
        self.c = c
        self.gamma = gamma
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.steps_per_input = steps_per_input
        self.boundary = boundary
        self.mode = mode.lower()
        self.device_str = device

        if use_optimal_step_size:
            self.steps_per_input = self.compute_steps_per_input(self.Nx, self.D, self.dt)

        self.state: Optional[torch.Tensor] = None
        self.velocity: Optional[torch.Tensor] = None

        # For sensor_positions (normalized [0,1])
        self.grid_x = np.linspace(0, 1, Nx)
        self.grid_y = np.linspace(0, 1, Ny)

        # Injection points (numpy RNG)
        rand_ix = np.random.randint(0, Nx, size=n_injection_points)
        rand_iy = np.random.randint(0, Ny, size=n_injection_points)
        injection_points = [(int(x), int(y)) for x, y in zip(rand_ix, rand_iy)]
        self.injection_points = torch.tensor(injection_points, dtype=torch.long)

        injection_dims = np.random.randint(0, input_size, size=n_injection_points)
        injection_scales = 5.0 * (np.random.rand(n_injection_points) - 0.5)
        self.injection_dims = torch.tensor(injection_dims, dtype=torch.long)
        self.injection_scales = torch.tensor(injection_scales, dtype=torch.float32)

        # Measurement points
        if use_injection_as_sensors:
            measurement_points = injection_points
        else:
            rand_ix_meas = np.random.randint(0, Nx, size=n_measurement_points)
            rand_iy_meas = np.random.randint(0, Ny, size=n_measurement_points)
            measurement_points = [(int(x), int(y)) for x, y in zip(rand_ix_meas, rand_iy_meas)]

        meas_pts_t = torch.tensor(measurement_points, dtype=torch.long)
        self.register_buffer("measurement_points", meas_pts_t)  # [n_meas,2]
        self.num_sensors = meas_pts_t.shape[0]

        sensor_xy = []
        for ix, iy in measurement_points:
            sensor_xy.append([self.grid_x[ix], self.grid_y[iy]])
        self.register_buffer(
            "sensor_positions", torch.from_numpy(np.asarray(sensor_xy, dtype=np.float32))
        )

        # Laplacian conv kernel
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        laplacian_kernel *= 1.0 / (dx * dx)
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)

        self.laplacian_conv = nn.Conv2d(1, 1, kernel_size=3, padding=0, bias=False)
        with torch.no_grad():
            self.laplacian_conv.weight.copy_(laplacian_kernel)
        self.laplacian_conv.weight.requires_grad_(False)

        self.pad_mode = "circular" if boundary == "periodic" else "constant"

        # forcing_mask: [input_size, Nx, Ny]
        forcing_mask_np = np.zeros((input_size, Nx, Ny), dtype=np.float32)
        for i in range(n_injection_points):
            d_i = injection_dims[i]
            sc_i = injection_scales[i]
            ix, iy = injection_points[i]
            forcing_mask_np[d_i, ix, iy] += sc_i
        self.register_buffer("forcing_mask", torch.from_numpy(forcing_mask_np))

        self.to(device)

    def compute_steps_per_input(self, Nx: int, D: float, dt: float) -> int:
        dx = 1.0 / (Nx - 1)
        optimal_step = int((dx * dx) / (4 * D * dt))
        return max(1, optimal_step)

    def reset_state(self, batch_size: int = 1, device: Optional[str] = None) -> None:
        if device is not None:
            self.to(device)
        else:
            self.to(self.device_str)

        dev = self.laplacian_conv.weight.device
        self.state = torch.zeros((self.Nx, self.Ny), dtype=torch.float32, device=dev)
        if self.mode == "wave":
            self.velocity = torch.zeros((self.Nx, self.Ny), dtype=torch.float32, device=dev)
        else:
            self.velocity = None

    def _build_forcing_sequence(self, forcing_field: torch.Tensor) -> List[torch.Tensor]:
        if self.forcing_type == "constant":
            return [forcing_field for _ in range(self.steps_per_input)]
        if self.forcing_type == "impulse":
            return [forcing_field] + [
                torch.zeros_like(forcing_field) for _ in range(self.steps_per_input - 1)
            ]
        if self.forcing_type == "periodic":
            period = self.steps_per_input * self.dt
            out = []
            for step in range(self.steps_per_input):
                t = step * self.dt
                factor = math.sin(2 * math.pi * t / period)
                out.append(forcing_field * factor)
            return out
        raise ValueError(f"Unknown forcing_type: {self.forcing_type}")

    def forward(self, u_t: torch.Tensor) -> torch.Tensor:
        batch_size, inp_dim = u_t.shape
        if batch_size != 1:
            raise ValueError("PDE2DReservoir expects batch_size=1")
        if inp_dim != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {inp_dim}")
        if self.state is None:
            self.reset_state(batch_size=1, device=str(u_t.device))

        inp_vec = u_t[0].view(-1, 1, 1)  # [input_size,1,1]
        forcing_field = (self.forcing_mask * inp_vec).sum(dim=0)  # [Nx,Ny]
        forces = self._build_forcing_sequence(forcing_field)

        self.state, self.velocity = _evolve_pde(
            self.state,
            self.velocity,
            forces,
            self.steps_per_input,
            self.mode,
            self.pad_mode,
            float(self.dt),
            float(self.D),
            float(self.alpha),
            float(self.c),
            float(self.gamma),
            self.laplacian_conv,
        )

        ixs = self.measurement_points[:, 0]
        iys = self.measurement_points[:, 1]
        measured = self.state[ixs, iys]  # [n_meas]
        return measured.unsqueeze(0)  # [1,n_meas]


def build_pde_reservoir_compiled(**kwargs) -> PDE2DReservoir:
    model = PDE2DReservoir(**kwargs)
    if HAS_TORCH_COMPILE:
        try:
            compiled_forward = torch_compile(model.forward)
            dummy = torch.zeros(1, model.input_size, device=model.laplacian_conv.weight.device)
            _ = compiled_forward(dummy)
            model.forward = compiled_forward
        except Exception:
            pass
    return model
