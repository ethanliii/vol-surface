"""Raw SVI smiles: evaluation, Durrleman's condition, and calibration.

Raw SVI (Gatheral 2004) parametrises total implied variance ``w = sigma^2 T``
as a function of log-forward-moneyness ``k = ln(K/F)``:

    w(k) = a + b [ rho (k - m) + sqrt((k - m)^2 + sigma^2) ]

with ``b >= 0``, ``|rho| < 1``, ``sigma > 0``. The smile is a hyperbola whose
wings are asymptotically linear with slopes ``b (1 - rho)`` (left) and
``b (1 + rho)`` (right). Roger Lee's moment formula bounds those slopes by 2,
and the minimum total variance ``a + b sigma sqrt(1 - rho^2)`` must be >= 0.

Calibration is done in two stages:

1. *Quasi-explicit* fit (Zeliade 2009, De Marco & Martini): for fixed
   ``(m, sigma)`` the model is linear in ``(a, u, v)`` with
   ``u = b sigma (1 + rho)`` and ``v = b sigma (1 - rho)``, and Lee's bound
   becomes the box ``0 <= u, v <= 2 sigma``. The inner problem is a bounded
   linear least-squares solve; only ``(m, sigma)`` is searched numerically.
   This stage enforces the parameter constraints but not butterfly or
   calendar no-arbitrage.

2. *Arbitrage-free* refit: starting from stage 1, SLSQP on all five
   parameters with inequality constraints ``g(k) >= 0`` (Durrleman) on a
   dense grid and, given the neighbouring fitted slice(s), total variance
   ordered in maturity on the grid plus ordered wing slopes (no calendar
   arbitrage). The surface module fits from the longest expiry backwards, so
   each slice is capped by the next longer one.

Residuals are measured in total variance and weighted by the inverse of each
quote's bid-ask uncertainty expressed in total variance, so the objective is
approximately ``sum(((iv_model - iv_mid) / iv_half_spread)^2)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import quadprog
from numpy.typing import ArrayLike
from scipy.optimize import lsq_linear, minimize

from volsurface.black_scholes import Array

LEE_MAX_SLOPE = 2.0
# Violations below these are numerical noise (1e-6 in total variance is about
# 0.0003 vol points at one year) and are not counted as arbitrage.
BUTTERFLY_TOL = 1e-6  # in units of Durrleman's g
CALENDAR_TOL = 1e-6  # in total variance


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def to_array(self) -> Array:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    @classmethod
    def from_array(cls, x: ArrayLike) -> SVIParams:
        a, b, rho, m, sigma = (float(v) for v in np.asarray(x, dtype=float))
        return cls(a, b, rho, m, sigma)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    @property
    def min_variance(self) -> float:
        return self.a + self.b * self.sigma * np.sqrt(1 - self.rho**2)

    def is_valid(self, tol: float = 1e-10) -> bool:
        return (
            self.b >= -tol
            and abs(self.rho) < 1
            and self.sigma > 0
            and self.min_variance >= -tol
            and self.b * (1 + abs(self.rho)) <= LEE_MAX_SLOPE + tol
        )


def svi_total_variance(p: SVIParams, k: ArrayLike) -> Array:
    x = np.asarray(k, dtype=float) - p.m
    return p.a + p.b * (p.rho * x + np.sqrt(x * x + p.sigma**2))


def svi_derivatives(p: SVIParams, k: ArrayLike) -> tuple[Array, Array, Array]:
    """(w, dw/dk, d2w/dk2)."""
    x = np.asarray(k, dtype=float) - p.m
    r = np.sqrt(x * x + p.sigma**2)
    w = p.a + p.b * (p.rho * x + r)
    w1 = p.b * (p.rho + x / r)
    w2 = p.b * p.sigma**2 / r**3
    return w, w1, w2


def durrleman_g(w: Array, w1: Array, w2: Array, k: ArrayLike) -> Array:
    """Durrleman's function; the slice is free of butterfly arbitrage iff g >= 0.

    g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2, and g(k) is (up to a
    positive factor) the risk-neutral density implied by the smile.
    """
    k = np.asarray(k, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1 - k * w1 / (2 * w)) ** 2 - (w1**2 / 4) * (1 / w + 0.25) + w2 / 2


def svi_g(p: SVIParams, k: ArrayLike) -> Array:
    w, w1, w2 = svi_derivatives(p, k)
    return durrleman_g(w, w1, w2, k)


def svi_implied_vol(p: SVIParams, k: ArrayLike, T: float) -> Array:
    return np.sqrt(np.maximum(svi_total_variance(p, k), 0) / T)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def constraint_grid(k: ArrayLike, theta: float, n: int = 201) -> Array:
    """Strike grid used to impose / check no-arbitrage: data range plus wings."""
    k = np.asarray(k, dtype=float)
    s = np.sqrt(max(theta, 1e-8))
    lo = min(k.min(), -6 * s)
    hi = max(k.max(), 6 * s)
    pad = 0.25 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, n)


def _design(k: Array, m: float, sigma: float) -> Array:
    """Columns multiplying (a, u, v): w = a + u (r + y)/2 + v (r - y)/2."""
    y = (k - m) / sigma
    r = np.sqrt(y * y + 1)
    return np.column_stack([np.ones_like(y), (r + y) / 2, (r - y) / 2])


def _uv_to_params(a: float, u: float, v: float, m: float, sigma: float) -> SVIParams:
    c = (u + v) / 2
    d = (u - v) / 2
    b = c / sigma
    rho = d / c if c > 0 else 0.0
    return SVIParams(float(a), float(b), float(np.clip(rho, -0.999999, 0.999999)), m, sigma)


def _slopes(p: SVIParams) -> Array:
    """(left, right) asymptotic slopes of total variance."""
    return np.array([p.b * (1 - p.rho), p.b * (1 + p.rho)])


@dataclass
class _Problem:
    """One slice's calibration problem, with total variance normalised by theta."""

    k: Array
    w: Array  # normalised by theta
    wt: Array  # normalised to mean 1
    theta: float
    bf_grid: Array | None  # butterfly grid (None: no butterfly constraint)
    cal_grid: Array
    lower: tuple[Array, Array] | None = None  # (w on cal_grid / theta, slopes)
    upper: tuple[Array, Array] | None = None

    def bounds(self, sigma: float) -> tuple[Array, Array]:
        """Box on (a, u, v). u / sigma and v / sigma are the wing slopes (times theta)."""
        s_max = LEE_MAX_SLOPE * sigma / self.theta
        lo = np.array([-2 * self.w.max() - 1, 0.0, 0.0])
        hi = np.array([2 * self.w.max() + 1, s_max, s_max])
        if self.lower is not None:  # wings at least as steep as the shorter slice
            left, right = self.lower[1]
            lo[1] = max(lo[1], right * sigma / self.theta)
            lo[2] = max(lo[2], left * sigma / self.theta)
        if self.upper is not None:  # ... and no steeper than the longer slice
            left, right = self.upper[1]
            hi[1] = min(hi[1], right * sigma / self.theta)
            hi[2] = min(hi[2], left * sigma / self.theta)
        return lo, hi

    def inner(self, m: float, sigma: float) -> tuple[Array, float, float]:
        """Weighted LS in (a, u, v) under box and calendar (linear) constraints.

        Returns (x, sse, calendar_violation). If no (a, u, v) satisfies the
        calendar constraints for this (m, sigma), the least-violating solution
        is returned and the violation is penalised by the outer search.
        """
        lo, hi = self.bounds(sigma)
        box_violation = float(np.sum(np.maximum(lo - hi, 0)))
        hi = np.maximum(hi, lo + 1e-12 * np.maximum(1.0, np.abs(lo)))
        A = _design(self.k, m, sigma)
        sw = np.sqrt(self.wt)
        x = lsq_linear(A * sw[:, None], self.w * sw, bounds=(lo, hi), method="bvls").x
        rows, rhs = [], []  # constraints rows @ x >= rhs
        if self.lower is not None or self.upper is not None:
            G = _design(self.cal_grid, m, sigma)
            if self.lower is not None:
                rows.append(G)
                rhs.append(self.lower[0])
            if self.upper is not None:
                rows.append(-G)
                rhs.append(-self.upper[0])
        cal_violation = box_violation
        if rows:
            C, d = np.vstack(rows), np.concatenate(rhs)
            if np.min(C @ x - d) < -1e-12:
                x = self._qp(A, sw, C, d, lo, hi, x)
            cal_violation += float(np.mean(np.maximum(d - C @ x, 0)))
        sse = float(np.mean(self.wt * (A @ x - self.w) ** 2))
        return x, sse, cal_violation

    def _qp(self, A, sw, C, d, lo, hi, x0) -> Array:
        """Convex QP (Goldfarb-Idnani): min |W^1/2 (A x - w)|^2 s.t. C x >= d, box."""
        Aw, bw = A * sw[:, None], self.w * sw
        H = Aw.T @ Aw
        H += 1e-12 * np.trace(H) * np.eye(3)  # strictly positive definite
        eye = np.eye(3)
        C_all = np.vstack([C, eye, -eye])
        d_all = np.concatenate([d, lo, -hi])
        try:
            return quadprog.solve_qp(H, Aw.T @ bw, C_all.T, d_all, 0)[0]
        except ValueError:  # inconsistent constraints: keep the box-only solution
            return x0

    def params(self, x: Array, m: float, sigma: float) -> SVIParams:
        a, u, v = x * self.theta
        return _uv_to_params(a, u, v, m, sigma)

    def objective(self, z: Array, bf_weight: float = 10.0) -> float:
        m, sigma = z[0], float(np.exp(z[1]))
        x, sse, cal_violation = self.inner(m, sigma)
        p = self.params(x, m, sigma)
        penalty = 100.0 * cal_violation
        min_var = p.min_variance / self.theta
        if min_var < 0:
            penalty += 10.0 * min_var**2 + 1e-3 * abs(min_var)
        if self.bf_grid is not None:
            g = svi_g(p, self.bf_grid)
            penalty += bf_weight * float(np.mean(np.maximum(-np.nan_to_num(g, nan=-1.0), 0)))
        return sse + penalty

    def solve(self) -> SVIParams:
        span = max(self.k.max() - self.k.min(), 1e-3)
        ms = np.linspace(self.k.min() - 0.1 * span, self.k.max() + 0.1 * span, 11)
        sigmas = np.geomspace(max(0.01 * span, 1e-4), 1.5 * span, 10)
        scored = sorted(
            (self.objective(np.array([m, np.log(s)])), m, s) for m in ms for s in sigmas
        )
        best = None
        for _, m0, s0 in scored[:3]:
            r = minimize(
                self.objective,
                np.array([m0, np.log(s0)]),
                method="Nelder-Mead",
                options={"xatol": 1e-6, "fatol": 1e-13, "maxiter": 600},
            )
            if best is None or r.fun < best.fun:
                best = r
        assert best is not None
        m, sigma = best.x[0], float(np.exp(best.x[1]))
        return self.params(self.inner(m, sigma)[0], m, sigma)


@dataclass
class FitInfo:
    feasible: bool
    max_butterfly_violation: float  # max(-g) on the butterfly grid, 0 if none
    max_calendar_violation: float  # in total variance, 0 if none
    polished: bool


def _normalise(k, w, weights):
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    wt = np.ones_like(w) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(k)
    theta = float(max(np.interp(0.0, k[order], w[order]), 1e-10))
    return k, w, wt / wt.mean(), theta


def fit_svi_raw(k: ArrayLike, w: ArrayLike, weights: ArrayLike | None = None) -> SVIParams:
    """Stage 1: quasi-explicit calibration with parameter constraints only
    (b >= 0, |rho| < 1, sigma > 0, Lee wing bound, non-negative variance)."""
    k, w, wt, theta = _normalise(k, w, weights)
    prob = _Problem(k, w / theta, wt, theta, bf_grid=None, cal_grid=k)
    return prob.solve()


def fit_svi_arbitrage_free(
    k: ArrayLike,
    w: ArrayLike,
    weights: ArrayLike | None = None,
    lower: SVIParams | None = None,
    upper: SVIParams | None = None,
    k_grid: ArrayLike | None = None,
) -> tuple[SVIParams, FitInfo]:
    """Stage 2: SVI slice free of butterfly arbitrage (Durrleman's g >= 0 on
    ``k_grid``) and of calendar arbitrage against its neighbouring slices
    (``lower <= w <= upper`` on ``k_grid``, wing slopes ordered likewise).

    Calendar constraints and slope ordering are linear in (a, u, v) and are
    imposed exactly in the inner problem. Durrleman's condition enters the
    outer (m, sigma) search as an exact L1 penalty; if a residual violation
    remains, a final SLSQP polish on all five parameters enforces it.
    """
    k, w, wt, theta = _normalise(k, w, weights)
    grid = constraint_grid(k, theta) if k_grid is None else np.asarray(k_grid, dtype=float)

    def neighbour(p: SVIParams | None):
        return None if p is None else (svi_total_variance(p, grid) / theta, _slopes(p))

    prob = _Problem(k, w / theta, wt, theta, grid, grid, neighbour(lower), neighbour(upper))
    p = prob.solve()
    info = _check(p, grid, lower, upper)
    if not info.feasible:
        p_pol = _polish(p, prob, lower, upper)
        info_pol = _check(p_pol, grid, lower, upper)
        if info_pol.feasible or (
            info_pol.max_butterfly_violation + info_pol.max_calendar_violation
            < info.max_butterfly_violation + info.max_calendar_violation
        ):
            p, info = p_pol, info_pol
            info.polished = True
    return p, info


def _check(p: SVIParams, grid: Array, lower: SVIParams | None, upper: SVIParams | None) -> FitInfo:
    bf = float(max(0.0, -np.min(svi_g(p, grid))))
    w = svi_total_variance(p, grid)
    cal = 0.0
    if lower is not None:
        cal = max(cal, float(np.max(svi_total_variance(lower, grid) - w)))
    if upper is not None:
        cal = max(cal, float(np.max(w - svi_total_variance(upper, grid))))
    ok = bf <= BUTTERFLY_TOL and cal <= CALENDAR_TOL and p.is_valid()
    return FitInfo(ok, bf, cal, False)


def _polish(
    p: SVIParams, prob: _Problem, lower: SVIParams | None, upper: SVIParams | None
) -> SVIParams:
    theta, grid = prob.theta, prob.bf_grid
    assert grid is not None

    def unpack(z: Array) -> SVIParams:
        return SVIParams(z[0] * theta, z[1] * theta, z[2], z[3], z[4])

    def objective(z: Array) -> float:
        r = svi_total_variance(unpack(z), prob.k) / theta - prob.w
        return float(np.mean(prob.wt * r * r))

    def cons(z: Array) -> Array:
        q = unpack(z)
        parts = [svi_g(q, grid), [q.min_variance / theta], LEE_MAX_SLOPE - _slopes(q)]
        wq = svi_total_variance(q, grid) / theta
        if prob.lower is not None:
            parts += [wq - prob.lower[0], _slopes(q) - prob.lower[1]]
        if prob.upper is not None:
            parts += [prob.upper[0] - wq, prob.upper[1] - _slopes(q)]
        return np.concatenate(parts)

    span = grid.max() - grid.min()
    bounds = [
        (-10.0, 10.0),
        (0.0, LEE_MAX_SLOPE / theta),
        (-0.999, 0.999),
        (grid.min(), grid.max()),
        (1e-5, 2 * span),
    ]
    z0 = np.array([p.a / theta, p.b / theta, p.rho, p.m, p.sigma])
    z0 = np.clip(z0, [b[0] for b in bounds], [b[1] for b in bounds])
    r = minimize(
        objective,
        z0,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": cons}],
        options={"maxiter": 500, "ftol": 1e-14},
    )
    return unpack(r.x)


def iv_weights(iv: Array, iv_bid: Array, iv_ask: Array, T: float, floor: float = 0.001) -> Array:
    """Weights 1 / err_w^2 with err_w the half-spread expressed in total variance.

    dw = 2 sigma T d sigma; the half-spread in vol is floored (0.1 vol point by
    default) so very tight quotes do not dominate the fit.
    """
    half = np.abs(iv_ask - iv_bid) / 2
    half = np.where(np.isfinite(half), half, np.nanmedian(half))
    err_w = 2 * iv * T * np.maximum(half, floor)
    return 1.0 / err_w**2


def fit_metrics(p: SVIParams, k: Array, w: Array, wt: Array, T: float) -> tuple[float, float]:
    model = svi_total_variance(p, k)
    theta = max(float(np.median(w)), 1e-12)
    wn = wt / wt.mean()
    loss = float(np.mean(wn * ((model - w) / theta) ** 2))
    rmse = float(np.sqrt(np.mean((np.sqrt(np.maximum(model, 0) / T) - np.sqrt(w / T)) ** 2)))
    return loss, rmse
