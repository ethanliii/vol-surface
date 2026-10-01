"""Black-Scholes(-Merton) and Black-76 pricing and Greeks, vectorized with numpy.

Two parametrisations are provided:

* Spot form (Black-Scholes-Merton) with continuous dividend yield ``q``:
  ``C = S e^{-qT} N(d1) - K e^{-rT} N(d2)``,
  ``d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))``, ``d2 = d1 - sigma sqrt(T)``.

* Forward form (Black-76), which is what the surface engine uses because the
  forward ``F`` and discount factor ``D`` are inferred from put-call parity:
  ``C = D [F N(d1) - K N(d2)]``, ``d1 = [-k + w/2] / sqrt(w)`` with
  log-moneyness ``k = ln(K/F)`` and total implied variance ``w = sigma^2 T``.

Dividing the forward-form price by ``D F`` gives the *normalized* price, which
depends only on ``k`` and total volatility ``s = sqrt(w)``. The implied
volatility solver works entirely in these normalized units.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

Array = NDArray[np.float64]
SQRT_2PI = np.sqrt(2.0 * np.pi)


def norm_pdf(x: ArrayLike) -> Array:
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / SQRT_2PI


def norm_cdf(x: ArrayLike) -> Array:
    return np.asarray(ndtr(np.asarray(x, dtype=float)))


def _sign(is_call: ArrayLike) -> Array:
    """+1 for calls, -1 for puts."""
    return np.where(np.asarray(is_call, dtype=bool), 1.0, -1.0)


# ---------------------------------------------------------------------------
# Normalized (forward, undiscounted, per unit of forward) prices
# ---------------------------------------------------------------------------


def normalized_price(k: ArrayLike, s: ArrayLike, is_call: ArrayLike = True) -> Array:
    """Undiscounted option price divided by the forward, as a function of
    log-moneyness ``k = ln(K/F)`` and total volatility ``s = sigma sqrt(T)``.

    call: N(d1) - e^k N(d2);  put: e^k N(-d2) - N(-d1);  d1 = -k/s + s/2.
    """
    k = np.asarray(k, dtype=float)
    s = np.asarray(s, dtype=float)
    eps = _sign(is_call)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = -k / s + 0.5 * s
    d2 = d1 - s
    price = eps * (norm_cdf(eps * d1) - np.exp(k) * norm_cdf(eps * d2))
    # s == 0: intrinsic value of the forward contract.
    intrinsic = np.maximum(eps * (1.0 - np.exp(k)), 0.0)
    return np.where(s > 0, price, intrinsic)


def normalized_vega(k: ArrayLike, s: ArrayLike) -> Array:
    """d(normalized price)/ds; identical for calls and puts: phi(d1)."""
    k = np.asarray(k, dtype=float)
    s = np.asarray(s, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = -k / s + 0.5 * s
    return np.where(s > 0, norm_pdf(d1), 0.0)


# ---------------------------------------------------------------------------
# Black-76 (forward form)
# ---------------------------------------------------------------------------


def black76_price(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    sigma: ArrayLike,
    D: ArrayLike = 1.0,
    is_call: ArrayLike = True,
) -> Array:
    F, K, T, sigma, D = (np.asarray(x, dtype=float) for x in (F, K, T, sigma, D))
    k = np.log(K / F)
    return D * F * normalized_price(k, sigma * np.sqrt(T), is_call)


def black76_vega(
    F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike, D: ArrayLike = 1.0
) -> Array:
    """dC/dsigma (per 1.00 of vol, not per vol point)."""
    F, K, T, sigma, D = (np.asarray(x, dtype=float) for x in (F, K, T, sigma, D))
    sqrt_t = np.sqrt(T)
    return D * F * normalized_vega(np.log(K / F), sigma * sqrt_t) * sqrt_t


def black76_delta(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    sigma: ArrayLike,
    D: ArrayLike = 1.0,
    is_call: ArrayLike = True,
) -> Array:
    """Discounted forward delta dC/dF."""
    F, K, T, sigma, D = (np.asarray(x, dtype=float) for x in (F, K, T, sigma, D))
    s = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    eps = _sign(is_call)
    return D * eps * norm_cdf(eps * d1)


# ---------------------------------------------------------------------------
# Black-Scholes-Merton (spot form) and Greeks
# ---------------------------------------------------------------------------


def _d1_d2(S, K, T, r, q, sigma):
    S, K, T, r, q, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    vol_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_t
    return S, K, T, r, q, sigma, d1, d1 - vol_t


def bs_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    is_call: ArrayLike = True,
) -> Array:
    S, K, T, r, q, sigma, d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    eps = _sign(is_call)
    return eps * (S * np.exp(-q * T) * norm_cdf(eps * d1) - K * np.exp(-r * T) * norm_cdf(eps * d2))


def bs_greeks(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    is_call: ArrayLike = True,
) -> dict[str, Array]:
    """Analytic Greeks. Units: vega and rho per 1.00 (not per 1%), theta per year
    (the derivative with respect to calendar time, i.e. -dV/dT)."""
    S, K, T, r, q, sigma, d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    eps = _sign(is_call)
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    pdf1 = norm_pdf(d1)
    sqrt_t = np.sqrt(T)
    vega = S * dq * pdf1 * sqrt_t
    return {
        "price": eps * (S * dq * norm_cdf(eps * d1) - K * dr * norm_cdf(eps * d2)),
        "delta": eps * dq * norm_cdf(eps * d1),
        "gamma": dq * pdf1 / (S * sigma * sqrt_t),
        "vega": vega,
        "theta": (
            -S * dq * pdf1 * sigma / (2 * sqrt_t)
            - eps * r * K * dr * norm_cdf(eps * d2)
            + eps * q * S * dq * norm_cdf(eps * d1)
        ),
        "rho": eps * K * T * dr * norm_cdf(eps * d2),
        "vanna": -dq * pdf1 * d2 / sigma,  # d delta / d sigma
        "volga": vega * d1 * d2 / sigma,  # d vega / d sigma
    }
