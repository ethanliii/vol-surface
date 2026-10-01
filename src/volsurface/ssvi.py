"""Surface SVI (Gatheral & Jacquier, 2014) with power-law curvature.

    w(k, theta) = theta / 2 * (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2))
    phi(theta)  = eta / (theta^gamma (1 + theta)^(1 - gamma))

``theta_t`` is the ATM total variance of expiry ``t``. The whole surface has
three parameters (rho, eta, gamma) plus the ATM term structure. By Gatheral &
Jacquier (Theorem 4.1, Remark 4.4) the surface is free of static arbitrage if

* ``theta_t`` is non-decreasing in ``t`` (calendar),
* ``0 < gamma <= 1/2`` and ``eta (1 + |rho|) <= 2`` (butterfly),

so both conditions are imposed as parameter constraints. ``theta_t`` is taken
from the market ATM total variance per expiry, made monotone by isotonic
regression.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from volsurface.black_scholes import Array
from volsurface.svi import durrleman_g


@dataclass(frozen=True)
class SSVIParams:
    rho: float
    eta: float
    gamma: float

    def phi(self, theta: ArrayLike) -> Array:
        theta = np.asarray(theta, dtype=float)
        return self.eta / (theta**self.gamma * (1 + theta) ** (1 - self.gamma))

    def satisfies_no_arbitrage(self) -> bool:
        return 0 < self.gamma <= 0.5 and self.eta * (1 + abs(self.rho)) <= 2 + 1e-12


def ssvi_total_variance(p: SSVIParams, k: ArrayLike, theta: ArrayLike) -> Array:
    k = np.asarray(k, dtype=float)
    theta = np.asarray(theta, dtype=float)
    pk = p.phi(theta) * k
    return 0.5 * theta * (1 + p.rho * pk + np.sqrt((pk + p.rho) ** 2 + 1 - p.rho**2))


def ssvi_derivatives(p: SSVIParams, k: ArrayLike, theta: float) -> tuple[Array, Array, Array]:
    k = np.asarray(k, dtype=float)
    phi = float(p.phi(theta))
    x = phi * k + p.rho
    r = np.sqrt(x * x + 1 - p.rho**2)
    w = 0.5 * theta * (1 + p.rho * phi * k + r)
    w1 = 0.5 * theta * phi * (p.rho + x / r)
    w2 = 0.5 * theta * phi**2 * (1 - p.rho**2) / r**3
    return w, w1, w2


def ssvi_g(p: SSVIParams, k: ArrayLike, theta: float) -> Array:
    w, w1, w2 = ssvi_derivatives(p, k, theta)
    return durrleman_g(w, w1, w2, k)


def isotonic_increasing(y: ArrayLike, weights: ArrayLike | None = None) -> Array:
    """Pool-adjacent-violators: closest non-decreasing sequence in weighted L2."""
    y = np.asarray(y, dtype=float)
    wts = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    blocks: list[list[float]] = []  # [value, weight, count]
    for yi, wi in zip(y, wts, strict=True):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return np.concatenate([[v] * int(n) for v, _, n in blocks])


@dataclass
class SSVISurface:
    params: SSVIParams
    T: Array  # expiries (years), increasing
    theta: Array  # monotone ATM total variance per expiry

    def theta_at(self, T: float) -> float:
        if self.T[0] >= T:
            return float(self.theta[0] * T / self.T[0])
        if self.T[-1] <= T:
            return float(self.theta[-1] * T / self.T[-1])
        return float(np.interp(T, self.T, self.theta))

    def total_variance(self, k: ArrayLike, T: float) -> Array:
        return ssvi_total_variance(self.params, k, self.theta_at(T))

    def implied_vol(self, k: ArrayLike, T: float) -> Array:
        return np.sqrt(self.total_variance(k, T) / T)


def fit_ssvi(
    ks: list[Array], ws: list[Array], weights: list[Array], T: Array, theta_mkt: Array
) -> SSVISurface:
    """Fit (rho, eta, gamma) to all slices jointly under the no-arbitrage constraints."""
    T = np.asarray(T, dtype=float)
    theta = isotonic_increasing(np.maximum(theta_mkt, 1e-10))
    norm_wts = [wt / wt.mean() for wt in weights]

    def unpack(z: Array) -> SSVIParams:
        return SSVIParams(float(z[0]), float(z[1]), float(z[2]))

    def objective(z: Array) -> float:
        p = unpack(z)
        tot, n = 0.0, 0
        for k, w, wt, th in zip(ks, ws, norm_wts, theta, strict=True):
            r = (ssvi_total_variance(p, k, th) - w) / th
            tot += float(np.sum(wt * r * r))
            n += len(k)
        return tot / n

    cons = [
        {"type": "ineq", "fun": lambda z: 2 - z[1] * (1 + z[0])},
        {"type": "ineq", "fun": lambda z: 2 - z[1] * (1 - z[0])},
    ]
    bounds = [(-0.999, 0.999), (1e-4, 2.0), (1e-3, 0.5)]
    best = None
    for z0 in ([-0.7, 1.0, 0.3], [-0.5, 0.5, 0.45], [-0.9, 1.0, 0.1], [0.0, 0.5, 0.25]):
        r = minimize(
            objective,
            np.array(z0),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-14},
        )
        if best is None or r.fun < best.fun:
            best = r
    assert best is not None
    return SSVISurface(unpack(best.x), T, theta)
