"""Volatility surface assembled from per-expiry SVI slices.

Fitting procedure (``fit_svi_surface``):

1. Raw SVI per expiry (quasi-explicit, parameter constraints only).
2. Arbitrage-free refit of every slice under Durrleman's condition and calendar
   constraints against its fitted neighbours in maturity. Slices are refit in
   order of strike coverage, widest first. Wide slices pin down the wings with
   data; narrow slices (e.g. quarterly series quoted over few strikes) are then
   sandwiched between already-fitted neighbours rather than letting their
   extrapolated wings constrain data-rich expiries. Adjacent fitted slices are
   always mutually consistent, so each insertion only needs its two nearest
   fitted neighbours.

Between expiries, total variance is interpolated linearly in maturity at fixed
log-moneyness, which preserves the calendar ordering of the slices. Before the
first expiry, ``w(k, T) = (T / T_1) w_1(k)``; beyond the last, implied vol is
held constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from volsurface.black_scholes import Array
from volsurface.svi import (
    FitInfo,
    SVIParams,
    constraint_grid,
    fit_metrics,
    fit_svi_arbitrage_free,
    fit_svi_raw,
    iv_weights,
    svi_total_variance,
)


@dataclass
class Slice:
    expiry: date
    T: float
    F: float
    k: Array
    w: Array
    iv: Array
    iv_bid: Array
    iv_ask: Array
    weights: Array
    raw: SVIParams | None = None
    free: SVIParams | None = None  # arbitrage-free fit
    info: FitInfo | None = None

    @property
    def theta_market(self) -> float:
        order = np.argsort(self.k)
        return float(np.interp(0.0, self.k[order], self.w[order]))


@dataclass
class SVISurface:
    slices: list[Slice]
    k_grid: Array
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> Array:
        return np.array([s.T for s in self.slices])

    def params(self, which: str = "free") -> list[SVIParams]:
        return [getattr(s, which) for s in self.slices]

    def slice_w(self, k: Array, which: str = "free") -> Array:
        """Total variance of each slice on k: shape (n_slices, len(k))."""
        return np.array([svi_total_variance(p, k) for p in self.params(which)])

    def total_variance(self, k: Array | float, T: float, which: str = "free") -> Array:
        k = np.atleast_1d(np.asarray(k, dtype=float))
        Ts = self.T
        W = self.slice_w(k, which)
        if Ts[0] >= T:
            return W[0] * T / Ts[0]
        if Ts[-1] <= T:
            return W[-1] * T / Ts[-1]
        j = int(np.searchsorted(Ts, T))
        lam = (T - Ts[j - 1]) / (Ts[j] - Ts[j - 1])
        return (1 - lam) * W[j - 1] + lam * W[j]

    def implied_vol(self, k: Array | float, T: float, which: str = "free") -> Array:
        return np.sqrt(np.maximum(self.total_variance(k, T, which), 0) / T)

    def atm_vol(self, T: float, which: str = "free") -> float:
        return float(self.implied_vol(0.0, T, which)[0])

    def summary(self) -> pd.DataFrame:
        rows = []
        for s in self.slices:
            _, rmse_r = fit_metrics(s.raw, s.k, s.w, s.weights, s.T)
            _, rmse_f = fit_metrics(s.free, s.k, s.w, s.weights, s.T)
            iv_raw = np.sqrt(np.maximum(svi_total_variance(s.raw, s.k), 0) / s.T)
            iv_free = np.sqrt(np.maximum(svi_total_variance(s.free, s.k), 0) / s.T)
            rows.append(
                {
                    "expiry": s.expiry,
                    "days": s.T * 365,
                    "n_quotes": len(s.k),
                    "k_min": s.k.min(),
                    "k_max": s.k.max(),
                    "atm_vol_market": np.sqrt(s.theta_market / s.T),
                    "atm_vol_fit": np.sqrt(svi_total_variance(s.free, 0.0) / s.T),
                    "rmse_vol_raw": rmse_r,
                    "rmse_vol_free": rmse_f,
                    "in_spread_raw": _in_spread(iv_raw, s),
                    "in_spread_free": _in_spread(iv_free, s),
                    "feasible": s.info.feasible if s.info else False,
                    **{f"{n}": v for n, v in s.free.as_dict().items()},
                }
            )
        return pd.DataFrame(rows)


def _in_spread(iv_model: Array, s: Slice) -> float:
    ok = np.isfinite(s.iv_bid) & np.isfinite(s.iv_ask)
    inside = (iv_model >= s.iv_bid - 1e-12) & (iv_model <= s.iv_ask + 1e-12)
    return float(inside[ok].mean()) if ok.any() else np.nan


def build_slices(quotes: pd.DataFrame, min_days: float = 2.0, min_quotes: int = 8) -> list[Slice]:
    used = quotes[quotes["used"]]
    slices = []
    for expiry, g in used.groupby("expiry", sort=True):
        T = float(g["T"].iloc[0])
        if min_days > T * 365 or len(g) < min_quotes:
            continue
        g = g.sort_values("k")
        iv, ivb, iva = (g[c].to_numpy(float) for c in ("iv", "iv_bid", "iv_ask"))
        slices.append(
            Slice(
                expiry=expiry,
                T=T,
                F=float(g["F"].iloc[0]),
                k=g["k"].to_numpy(float),
                w=g["w"].to_numpy(float),
                iv=iv,
                iv_bid=ivb,
                iv_ask=iva,
                weights=iv_weights(iv, ivb, iva, T),
            )
        )
    return slices


def global_grid(slices: list[Slice], n: int = 301) -> Array:
    lo = min(constraint_grid(s.k, s.theta_market)[0] for s in slices)
    hi = max(constraint_grid(s.k, s.theta_market)[-1] for s in slices)
    return np.linspace(lo, hi, n)


def fit_svi_surface(slices: list[Slice], n_grid: int = 301) -> SVISurface:
    if not slices:
        raise ValueError("no slices to fit")
    slices = sorted(slices, key=lambda s: s.T)
    grid = global_grid(slices, n_grid)
    for s in slices:
        s.raw = fit_svi_raw(s.k, s.w, s.weights)

    order = sorted(range(len(slices)), key=lambda i: -(slices[i].k.max() - slices[i].k.min()))
    fitted: list[int] = []
    for i in order:
        lower_idx = max((j for j in fitted if j < i), default=None)
        upper_idx = min((j for j in fitted if j > i), default=None)
        s = slices[i]
        k_grid = np.union1d(constraint_grid(s.k, s.theta_market), grid)
        s.free, s.info = fit_svi_arbitrage_free(
            s.k,
            s.w,
            s.weights,
            lower=slices[lower_idx].free if lower_idx is not None else None,
            upper=slices[upper_idx].free if upper_idx is not None else None,
            k_grid=k_grid,
        )
        fitted.append(i)
    return SVISurface(slices, grid)
