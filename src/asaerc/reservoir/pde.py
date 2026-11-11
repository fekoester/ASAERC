from __future__ import annotations

import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch import compile as torch_compile  # torch>=2

    HAS_TORCH_COMPILE = True
except Exception:
    HAS_TORCH_COMPILE = False


def _laplacian_2d(U: torch.Tensor, inv_dx2: float, inv_dy2: float, pad_mode: str) -> torch.Tensor:
    """5-point Laplacian of U ([..., Nx, Ny]) with separate x/y spacing.

    Periodic boundaries use ``torch.roll``; Dirichlet (zero) boundaries pad once and slice. Both are
    cheaper than the previous pad + conv2d per micro-step and also work on batched fields.
    """
    if pad_mode == "circular":
        lap_x = torch.roll(U, 1, dims=-2) + torch.roll(U, -1, dims=-2) - 2.0 * U
        lap_y = torch.roll(U, 1, dims=-1) + torch.roll(U, -1, dims=-1) - 2.0 * U
    else:
        P = F.pad(U, (1, 1, 1, 1), mode="constant", value=0.0)
        lap_x = P[..., 2:, 1:-1] + P[..., :-2, 1:-1] - 2.0 * U
        lap_y = P[..., 1:-1, 2:] + P[..., 1:-1, :-2] - 2.0 * U
    return lap_x * inv_dx2 + lap_y * inv_dy2


def forcing_factors(forcing_type: str, steps: int, dt: float) -> list[float]:
    """Per-micro-step multiplier of the input forcing field."""
    if forcing_type == "constant":
        return [1.0] * steps
    if forcing_type == "impulse":
        return [1.0] + [0.0] * (steps - 1)
    if forcing_type == "periodic":
        period = steps * dt
        return [math.sin(2 * math.pi * (step * dt) / period) for step in range(steps)]
    raise ValueError(f"Unknown forcing_type: {forcing_type}")


@torch.no_grad()
def _evolve_pde(
    U: torch.Tensor,
    V: torch.Tensor | None,
    forcing_field: torch.Tensor,
    factors: list[float],
    mode: str,
    pad_mode: str,
    dt: float,
    D: float,
    alpha: float,
    c: float,
    gamma: float,
    inv_dx2: float,
    inv_dy2: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Explicit micro-step loop (forward Euler / symplectic Euler for the wave equation)."""
    for factor in factors:
        f_i = forcing_field * factor if factor != 1.0 else forcing_field
        if mode == "diffusion":
            U = U + dt * (D * _laplacian_2d(U, inv_dx2, inv_dy2, pad_mode) + f_i)
        elif mode == "diffusion_leaky":
            U = alpha * U + dt * (D * _laplacian_2d(U, inv_dx2, inv_dy2, pad_mode) + f_i)
        elif mode == "wave":
            if V is None:
                V = torch.zeros_like(U)
            accel = c**2 * _laplacian_2d(U, inv_dx2, inv_dy2, pad_mode) - gamma * V + f_i
            V = V + dt * accel
            U = U + dt * V
        else:
            raise ValueError(f"Unknown PDE mode: {mode}")
    return U, V


def stability_number(mode: str, D: float, c: float, dt: float, dx: float, dy: float) -> float:
    """Explicit-scheme stability number; the scheme is stable for values <= 1.

    diffusion: 2 * D * dt * (1/dx^2 + 1/dy^2)   (i.e. D*dt/dx^2 <= 1/4 on a square grid)
    wave:      c * dt * sqrt(1/dx^2 + 1/dy^2)   (CFL condition)
    """
    inv = 1.0 / dx**2 + 1.0 / dy**2
    if mode == "wave":
        return float(c * dt * math.sqrt(inv))
    return float(2.0 * D * dt * inv)


class PDE2DReservoir(nn.Module):
    """
    2D PDE reservoir driven by inputs injected at random points.

      - forcing_mask built from random injection points (``seed`` for a private RNG; without it
        NumPy's global RNG is used, as in earlier versions)
      - partial sensor measurements by fancy indexing
      - batch_size >= 1: ``state`` is [Nx, Ny] for a single trajectory, [B, Nx, Ny] for a batch of
        independent trajectories (e.g. several input files simulated at once)
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
        seed: int | None = None,
        strict_stability: bool = False,
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

        self.stability = stability_number(self.mode, D, c, dt, dx, dy)
        if self.mode in {"diffusion", "diffusion_leaky", "wave"} and self.stability > 1.0:
            message = (
                f"Explicit {self.mode} scheme is unstable: stability number {self.stability:.3f} > 1 "
                f"(D={D}, c={c}, dt={dt}, dx={dx}, dy={dy}). Reduce dt or D/c, or increase dx."
            )
            if strict_stability:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        rng = np.random.default_rng(seed) if seed is not None else np.random

        self.state: torch.Tensor | None = None
        self.velocity: torch.Tensor | None = None

        # For sensor_positions (normalized [0,1])
        self.grid_x = np.linspace(0, 1, Nx)
        self.grid_y = np.linspace(0, 1, Ny)

        # Injection points (numpy RNG)
        randint = rng.integers if seed is not None else rng.randint
        rand_ix = randint(0, Nx, size=n_injection_points)
        rand_iy = randint(0, Ny, size=n_injection_points)
        injection_points = [(int(x), int(y)) for x, y in zip(rand_ix, rand_iy)]
        self.injection_points = torch.tensor(injection_points, dtype=torch.long)

        injection_dims = randint(0, input_size, size=n_injection_points)
        injection_scales = 5.0 * (rng.random(n_injection_points) - 0.5)
        self.injection_dims = torch.tensor(injection_dims, dtype=torch.long)
        self.injection_scales = torch.tensor(injection_scales, dtype=torch.float32)

        # Measurement points
        if use_injection_as_sensors:
            measurement_points = injection_points
        else:
            rand_ix_meas = randint(0, Nx, size=n_measurement_points)
            rand_iy_meas = randint(0, Ny, size=n_measurement_points)
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

        self.inv_dx2 = 1.0 / (dx * dx)
        self.inv_dy2 = 1.0 / (dy * dy)

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
        """Micro-steps per input so diffusion spreads about one grid cell (uses the configured dx)."""
        optimal_step = int((self.dx * self.dx) / (4 * D * dt))
        return max(1, optimal_step)

    def reset_state(self, batch_size: int = 1, device: str | None = None) -> None:
        """Zeros the field. ``batch_size > 1`` simulates independent trajectories in parallel."""
        self.to(device if device is not None else self.device_str)
        dev = self.forcing_mask.device
        shape = (self.Nx, self.Ny) if batch_size == 1 else (batch_size, self.Nx, self.Ny)
        self.state = torch.zeros(shape, dtype=torch.float32, device=dev)
        self.velocity = torch.zeros_like(self.state) if self.mode == "wave" else None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints from <=0.1 stored the (fixed) Laplacian conv kernel; it is no longer a module.
        state_dict.pop(prefix + "laplacian_conv.weight", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def batch_size(self) -> int:
        if self.state is None or self.state.dim() == 2:
            return 1
        return int(self.state.shape[0])

    def forward(self, u_t: torch.Tensor) -> torch.Tensor:
        """Advances one input step. u_t: [B, input_size] -> sensor values [B, n_meas]."""
        batch_size, inp_dim = u_t.shape
        if inp_dim != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {inp_dim}")
        if self.state is None or self.batch_size != batch_size:
            self.reset_state(batch_size=batch_size, device=str(u_t.device))

        # [B, input, 1, 1] * [input, Nx, Ny] -> [B, Nx, Ny]
        forcing_field = torch.einsum("bi,ixy->bxy", u_t.float(), self.forcing_mask)
        if batch_size == 1:
            forcing_field = forcing_field[0]

        self.state, self.velocity = _evolve_pde(
            self.state,
            self.velocity,
            forcing_field,
            forcing_factors(self.forcing_type, self.steps_per_input, float(self.dt)),
            self.mode,
            self.pad_mode,
            float(self.dt),
            float(self.D),
            float(self.alpha),
            float(self.c),
            float(self.gamma),
            self.inv_dx2,
            self.inv_dy2,
        )

        ixs = self.measurement_points[:, 0]
        iys = self.measurement_points[:, 1]
        measured = self.state[..., ixs, iys]  # [n_meas] or [B, n_meas]
        return measured.unsqueeze(0) if measured.dim() == 1 else measured


def build_pde_reservoir_compiled(**kwargs) -> PDE2DReservoir:
    model = PDE2DReservoir(**kwargs)
    if HAS_TORCH_COMPILE:
        try:
            compiled_forward = torch_compile(model.forward)
            dummy = torch.zeros(1, model.input_size, device=model.forcing_mask.device)
            _ = compiled_forward(dummy)
            model.forward = compiled_forward
        except Exception:
            pass
    return model
