import numpy as np
import pytest
import torch

from asaerc import AttentionReadout2DPDE, PDE2DReservoir
from asaerc.diagnostics.corr import compute_correlation_mats
from asaerc.experiment import evaluate_readout, forecast_vpt, simulate_dataset, train_readout
from asaerc.forecast import closed_loop_forecast, valid_prediction_time


@pytest.fixture
def cfg(tmp_path):
    from preprocess_data import preprocess_and_save
    from run_experiment import Config

    raw = tmp_path / "raw"
    raw.mkdir()
    t = np.linspace(0, 40, 300)
    np.save(raw / "sine_data.npy", np.stack([np.sin(t), np.cos(t)], axis=1))
    np.save(raw / "saw_data.npy", np.stack([(t % 3) / 3.0, np.sin(2 * t)], axis=1))
    preprocess_and_save(str(raw), str(tmp_path / "processed"))
    return Config(
        processed_dir=str(tmp_path / "processed"),
        input_dim=2,
        output_dim=2,
        Nx=16,
        Ny=16,
        steps_per_input=5,
        n_injection_points=40,
        n_measurement_points=12,
        D=1e-4,
        dx=0.002,
        dy=0.002,
        dt=0.002,
        boundary="periodic",
        hidden_dim=16,
        n_queries=8,
        epochs=2,
        lr=1e-2,
    )


def test_preprocessing_uses_training_statistics(cfg):
    data = np.load(f"{cfg.processed_dir}/sine_data_std.npy")
    cutoff = int(0.8 * (len(data) - 1)) + 1
    np.testing.assert_allclose(data[:cutoff].mean(0), 0, atol=1e-10)
    assert np.abs(data[cutoff:].mean(0)).max() > 1e-3


@pytest.mark.parametrize("kind", ["linear", "static", "adaptive"])
@pytest.mark.parametrize("target_mode", ["next", "delta"])
def test_train_evaluate_forecast(cfg, kind, target_mode):
    torch.manual_seed(0)
    pde_res, splits = simulate_dataset(cfg, "cpu")
    assert [s.name for s in splits] == ["saw", "sine"]
    readout = train_readout(cfg, splits, kind, "cpu", target_mode=target_mode)
    scores = evaluate_readout(readout, splits, "cpu")
    assert set(scores) == {"saw", "sine"}
    assert all(np.isfinite(v["mse"]) for v in scores.values())
    vpt = forecast_vpt(pde_res, readout, splits, "cpu", warmup=10, steps=5)
    assert set(vpt) == {"saw", "sine"}


@pytest.mark.parametrize("linear", [True, False])
def test_correlation_diagnostics(cfg, linear):
    res = PDE2DReservoir(
        input_size=2,
        n_injection_points=20,
        n_measurement_points=6,
        Nx=12,
        Ny=12,
        D=1e-4,
        dx=0.002,
        dy=0.002,
        dt=0.002,
        steps_per_input=3,
        device="cpu",
        seed=0,
        use_injection_as_sensors=False,
    )
    readout = AttentionReadout2DPDE(
        num_sensors=6, output_dim=2, n_queries=5, hidden_dim=8, use_linear_model=linear
    )
    inputs = np.random.default_rng(0).normal(size=(40, 2)).astype(np.float32)
    c_plain, c_weighted, c_weights = compute_correlation_mats(res, readout, inputs, "cpu")
    n = 6 if linear else 5
    assert c_plain.shape == c_weighted.shape == c_weights.shape == (n, n)


def test_closed_loop_with_identity_like_readout():
    res = PDE2DReservoir(
        input_size=2,
        n_injection_points=20,
        n_measurement_points=6,
        Nx=12,
        Ny=12,
        D=1e-4,
        dx=0.002,
        dy=0.002,
        dt=0.002,
        steps_per_input=3,
        device="cpu",
        seed=0,
        use_injection_as_sensors=False,
    )

    class Zero(torch.nn.Module):
        def forward(self, meas, field):
            return torch.zeros(meas.shape[0], 2)

    warm = np.ones((5, 2), np.float32)
    assert np.allclose(closed_loop_forecast(res, Zero(), warm, 4, "cpu"), 0.0)
    # delta readout predicting zero change keeps the last input
    assert np.allclose(closed_loop_forecast(res, Zero(), warm, 4, "cpu", delta=True), 1.0)


def test_valid_prediction_time():
    truth = np.ones((10, 2))
    pred = truth.copy()
    pred[7:] += 3
    assert valid_prediction_time(pred, truth) == 7
