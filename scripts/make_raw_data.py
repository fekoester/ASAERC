from __future__ import annotations

import argparse
import os
from typing import Callable, List

import numpy as np
from scipy.integrate import solve_ivp


# ----------------------------
# ODE helper
# ----------------------------
def solve_ode_system(
    derivs_func: Callable,
    initial_state,
    n_steps: int,
    dt: float,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> np.ndarray:
    """
    Integrate an ODE using solve_ivp, returning array [n_steps, dim] at equally spaced t_eval.
    """
    T = n_steps * dt
    t_eval = np.linspace(0, T, n_steps)

    sol = solve_ivp(
        fun=derivs_func,
        t_span=(0, T),
        y0=np.asarray(initial_state, dtype=float),
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    return sol.y.T


# ----------------------------
# Systems
# ----------------------------
class DynamicalSystem:
    def __init__(self, name: str, dim: int):
        self.name = name
        self.dim = dim

    def generate_data(self, n_steps: int, dt: float) -> np.ndarray:
        raise NotImplementedError


class LorenzSystem(DynamicalSystem):
    def __init__(self, sigma=10.0, beta=8 / 3, rho=28.0):
        super().__init__(name="lorenz", dim=3)
        self.sigma, self.beta, self.rho = sigma, beta, rho

    def _odefunc(self, t, xyz):
        x, y, z = xyz
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return [dx, dy, dz]

    def generate_data(self, n_steps, dt):
        return solve_ode_system(self._odefunc, [1.0, 1.0, 1.0], n_steps, dt)


class RoesslerSystem(DynamicalSystem):
    def __init__(self, a=0.2, b=0.2, c=5.7):
        super().__init__(name="roessler", dim=3)
        self.a, self.b, self.c = a, b, c

    def _odefunc(self, t, xyz):
        x, y, z = xyz
        dx = -y - z
        dy = x + self.a * y
        dz = self.b + z * (x - self.c)
        return [dx, dy, dz]

    def generate_data(self, n_steps, dt):
        return solve_ode_system(self._odefunc, [0.0, 1.0, 1.0], n_steps, dt)


class VanDerPol(DynamicalSystem):
    def __init__(self, mu=3.0):
        super().__init__(name="vanderpol", dim=2)
        self.mu = mu

    def _odefunc(self, t, s):
        x, y = s
        dx = y
        dy = self.mu * (1 - x**2) * y - x
        return [dx, dy]

    def generate_data(self, n_steps, dt):
        return solve_ode_system(self._odefunc, [1.0, 0.0], n_steps, dt)


class Duffing(DynamicalSystem):
    def __init__(self, alpha=1.0, beta=-1.0, delta=0.2, F=0.3, omega=1.2):
        super().__init__(name="duffing", dim=2)
        self.alpha, self.beta, self.delta, self.F, self.omega = alpha, beta, delta, F, omega

    def _odefunc(self, t, s):
        x, y = s
        dx = y
        dy = -self.delta * y - self.alpha * x - self.beta * (x**3) + self.F * np.cos(self.omega * t)
        return [dx, dy]

    def generate_data(self, n_steps, dt):
        return solve_ode_system(self._odefunc, [0.1, 0.0], n_steps, dt)


class DoublePendulum(DynamicalSystem):
    def __init__(self, g=9.81, L1=1.0, L2=1.0, m1=1.0, m2=1.0):
        super().__init__(name="double_pendulum", dim=4)
        self.g, self.L1, self.L2, self.m1, self.m2 = g, L1, L2, m1, m2

    def _odefunc(self, t, s):
        th1, w1, th2, w2 = s
        m1, m2, L1, L2, g = self.m1, self.m2, self.L1, self.L2, self.g

        delta = th2 - th1
        den1 = (m1 + m2) * L1 - m2 * L1 * (np.cos(delta) ** 2)
        den2 = (L2 / L1) * den1

        dw1 = (
            m2 * L1 * (w1**2) * np.sin(delta) * np.cos(delta)
            + m2 * g * np.sin(th2) * np.cos(delta)
            + m2 * L2 * (w2**2) * np.sin(delta)
            - (m1 + m2) * g * np.sin(th1)
        ) / den1

        dw2 = (
            -m2 * L2 * (w2**2) * np.sin(delta) * np.cos(delta)
            + (m1 + m2) * g * np.sin(th1) * np.cos(delta)
            - (m1 + m2) * L1 * (w1**2) * np.sin(delta)
            - (m1 + m2) * g * np.sin(th2)
        ) / den2

        return [w1, dw1, w2, dw2]

    def generate_data(self, n_steps, dt):
        init_state = [np.pi / 2, 0.0, np.pi / 2 + 0.01, 0.0]
        return solve_ode_system(self._odefunc, init_state, n_steps, dt)


# Discrete maps
class MapSystem(DynamicalSystem):
    def __init__(self, name, dim):
        super().__init__(name, dim)

    def _map(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _initial_state(self) -> np.ndarray:
        return np.zeros((self.dim,), dtype=float)

    def generate_data(self, n_steps: int, dt: float = 1.0) -> np.ndarray:
        data = np.zeros((n_steps, self.dim))
        state = self._initial_state()
        for i in range(n_steps):
            state = self._map(state)
            data[i] = state
        return data


class LogisticMap(MapSystem):
    def __init__(self, r=3.99):
        super().__init__("logistic_map", dim=1)
        self.r = r

    def _initial_state(self):
        return np.array([0.5], dtype=float)

    def _map(self, x):
        return np.array([self.r * x[0] * (1 - x[0])], dtype=float)


class HenonMap(MapSystem):
    def __init__(self, a=1.4, b=0.3):
        super().__init__("henon_map", dim=2)
        self.a, self.b = a, b

    def _initial_state(self):
        return np.array([0.0, 0.0], dtype=float)

    def _map(self, x):
        x_n, y_n = x
        x_next = 1 - self.a * (x_n**2) + y_n
        y_next = self.b * x_n
        return np.array([x_next, y_next], dtype=float)


# Delay-based
class MackeyGlassSystem(DynamicalSystem):
    def __init__(self, beta=0.2, gamma=0.1, n=10, tau=17, x0=1.2):
        super().__init__(name="mackey_glass", dim=1)
        self.beta, self.gamma, self.n, self.tau, self.x0 = beta, gamma, n, tau, x0

    def generate_data(self, n_steps: int, dt: float = 0.1) -> np.ndarray:
        delay_steps = int(self.tau // dt)
        buffer = np.ones(delay_steps) * self.x0
        x = self.x0
        data = np.zeros((n_steps, 1))

        for i in range(n_steps):
            x_tau = buffer[i % delay_steps]
            dxdt = self.beta * x_tau / (1 + x_tau**self.n) - self.gamma * x
            x = x + dt * dxdt
            data[i, 0] = x
            buffer[i % delay_steps] = x

        return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="raw_data")
    ap.add_argument("--n_steps", type=int, default=5500)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--burn_in", type=int, default=500)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    systems: List[DynamicalSystem] = [
        LorenzSystem(),
        RoesslerSystem(),
        VanDerPol(mu=3.0),
        Duffing(),
        DoublePendulum(),
        LogisticMap(r=3.99),
        HenonMap(a=1.4, b=0.3),
        MackeyGlassSystem(beta=0.2, gamma=0.1, n=10, tau=17, x0=1.2),
    ]

    if args.plot:
        import matplotlib.pyplot as plt

        cols = 3
        rows = (len(systems) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        axes = axes.flatten()

    for idx, sys in enumerate(systems):
        if isinstance(sys, MapSystem):
            data = sys.generate_data(args.n_steps, dt=1.0)
        elif isinstance(sys, MackeyGlassSystem):
            data = sys.generate_data(args.n_steps, dt=0.1)
        else:
            data = sys.generate_data(args.n_steps, args.dt)

        if args.burn_in > 0:
            data = data[args.burn_in :]

        np.save(os.path.join(args.out_dir, f"{sys.name}_data.npy"), data)

        if args.plot:
            ax = axes[idx]
            ax.plot(data[:, 0], label=sys.name)
            ax.set_title(sys.name)
            ax.legend()

    if args.plot:
        import matplotlib.pyplot as plt

        plt.tight_layout()
        plt.show()

    print(f"Raw data saved to '{args.out_dir}/'")


if __name__ == "__main__":
    main()
