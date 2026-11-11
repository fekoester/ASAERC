import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from asaerc import PDE2DReservoir
from asaerc.reservoir.pde import _laplacian_2d, stability_number
from asaerc.simulation import simulate_pde_for_file, simulate_pde_for_files

KW = dict(
    input_size=3,
    n_injection_points=50,
    n_measurement_points=16,
    use_injection_as_sensors=False,
    Nx=20,
    Ny=24,
    D=1e-4,
    dx=0.002,
    dy=0.002,
    dt=0.002,
    steps_per_input=5,
    device="cpu",
    forcing_type="constant",
    mode="diffusion",
)


def _conv_laplacian(U, dx, pad_mode):
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    padded = F.pad(U[None, None], (1, 1, 1, 1), mode=pad_mode)
    return F.conv2d(padded, kernel)[0, 0] / dx**2


@pytest.mark.parametrize("pad_mode", ["circular", "constant"])
def test_laplacian_matches_conv_reference(pad_mode):
    U = torch.randn(20, 24)
    fast = _laplacian_2d(U, 1 / 0.1**2, 1 / 0.1**2, pad_mode)
    torch.testing.assert_close(fast, _conv_laplacian(U, 0.1, pad_mode), rtol=1e-5, atol=1e-4)


def test_laplacian_uses_dy():
    x = torch.linspace(0, 1, 30)
    U = (x[None, :] ** 2).repeat(30, 1)  # u = y^2 -> u_yy = 2
    lap = _laplacian_2d(U, 1.0, 1.0 / (x[1] - x[0]) ** 2, "constant")
    torch.testing.assert_close(lap[5:-5, 5:-5], torch.full((20, 20), 2.0), rtol=1e-3, atol=1e-3)


def test_stability_warning_and_strict_mode():
    assert stability_number("diffusion", 1e-4, 0.0, 0.002, 0.002, 0.002) == pytest.approx(0.2)
    with pytest.warns(RuntimeWarning):
        PDE2DReservoir(**{**KW, "D": 1e-2})
    with pytest.raises(ValueError):
        PDE2DReservoir(**{**KW, "D": 1e-2, "strict_stability": True})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PDE2DReservoir(**KW)


def test_seeded_reservoirs_are_reproducible():
    a, b, c = (PDE2DReservoir(**KW, seed=s) for s in (1, 1, 2))
    assert torch.equal(a.forcing_mask, b.forcing_mask)
    assert not torch.equal(a.forcing_mask, c.forcing_mask)
    np.random.seed(0)
    d = PDE2DReservoir(**KW)
    np.random.seed(0)
    e = PDE2DReservoir(**KW)
    assert torch.equal(d.forcing_mask, e.forcing_mask)  # legacy global-RNG path


@pytest.mark.parametrize("mode", ["diffusion", "diffusion_leaky", "wave"])
def test_batched_simulation_matches_sequential(mode):
    res = PDE2DReservoir(**{**KW, "mode": mode, "alpha": 0.95, "c": 0.1, "gamma": 0.1}, seed=0)
    rng = np.random.default_rng(0)
    series = [
        (rng.normal(size=(30, 3)).astype(np.float32), np.zeros((30, 3), np.float32))
        for _ in range(3)
    ]
    batched = simulate_pde_for_files(res, series, "cpu", field_dtype=torch.float32)
    for (inputs, targets), (meas, field, _) in zip(series, batched):
        meas_1, field_1, _ = simulate_pde_for_file(
            res, inputs, targets, "cpu", field_dtype=torch.float32
        )
        torch.testing.assert_close(meas, meas_1)
        torch.testing.assert_close(field, field_1)


def test_old_checkpoints_with_laplacian_kernel_load():
    res = PDE2DReservoir(**KW, seed=0)
    state = res.state_dict()
    state["laplacian_conv.weight"] = torch.zeros(1, 1, 3, 3)
    PDE2DReservoir(**KW, seed=1).load_state_dict(state)
