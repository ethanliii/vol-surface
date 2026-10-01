"""Vectorized implied-volatility solver.

Everything is done in normalized Black-76 units: given an undiscounted price
per unit of forward ``p`` and log-moneyness ``k``, find total volatility
``s = sigma sqrt(T)`` such that ``normalized_price(k, s) = p``.

Algorithm (per option, vectorized across the whole chain):

1. Convert in-the-money quotes to the equivalent out-of-the-money quote with
   put-call parity (``c - p = 1 - e^k`` in normalized units). OTM prices carry
   no intrinsic value, so the inversion is well conditioned.
2. Check no-arbitrage bounds: ``0 < p_otm < min(1, e^k)``. Prices outside the
   bounds (or ITM prices whose time value is below floating-point resolution)
   have no implied vol and return NaN.
3. Newton iterations on ``log(price)``. Working in log space keeps Newton
   effective in the far wings, where the price is tiny and convex in ``s``.
4. Safeguard: every iterate keeps a bracket ``[lo, hi]`` that contains the
   root (the price is strictly increasing in ``s``). If a Newton step would
   leave the bracket, or vega underflows, the step is replaced by bisection.
   The method therefore always converges, and converges quadratically once
   Newton takes over.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from volsurface.black_scholes import Array, normalized_price, normalized_vega

S_MAX = 10.0  # total vol cap: sigma = 1000% at T = 1y


def implied_total_vol(
    price: ArrayLike,
    k: ArrayLike,
    is_call: ArrayLike = True,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> Array:
    """Total implied volatility ``s`` from normalized price (undiscounted / F)."""
    price, k = np.broadcast_arrays(np.asarray(price, float), np.asarray(k, float))
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), price.shape)
    price = price.astype(float).copy()
    k = k.astype(float).copy()

    # 1. OTM-equivalent price via parity: call - put = 1 - e^k.
    otm_call = k >= 0
    parity = 1.0 - np.exp(k)
    p = np.where(is_call == otm_call, price, price - np.where(is_call, parity, -parity))

    # 2. Bounds: OTM call in (0, 1), OTM put in (0, e^k). An ITM quote whose
    # time value is lost in floating-point rounding of the quoted price carries
    # no vol information either.
    upper = np.where(otm_call, 1.0, np.exp(k))
    valid = np.isfinite(p) & np.isfinite(k) & (p > 0) & (p < upper)
    valid &= (p > 1e-13 * np.abs(price)) & (p > 1e-250)  # 1e-250: subnormal guard

    s = np.full(p.shape, np.nan)
    if not valid.any():
        return s
    pv, kv, cv = p[valid], k[valid], otm_call[valid]

    lo = np.zeros_like(pv)
    hi = np.full_like(pv, S_MAX)
    # Initial guess: point of maximum vega in s for fixed k, blended with the
    # Brenner-Subrahmanyam ATM approximation s ~ sqrt(2 pi) p.
    x = np.clip(np.maximum(np.sqrt(2.0 * np.abs(kv)), np.sqrt(2 * np.pi) * pv), 1e-4, S_MAX)
    log_target = np.log(pv)
    done = np.zeros_like(pv, dtype=bool)

    for _ in range(max_iter):
        model = normalized_price(kv, x, cv)
        f = model - pv
        # Maintain the bracket (price is increasing in s).
        lo = np.where(f < 0, x, lo)
        hi = np.where(f > 0, x, hi)
        done |= np.abs(f) <= tol * np.maximum(pv, 1e-300)
        done |= (hi - lo) <= 1e-15
        if done.all():
            break
        vega = normalized_vega(kv, x)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # Newton on g(s) = log(price(s)) - log(target): g' = vega / price.
            step = (np.log(np.maximum(model, 1e-300)) - log_target) * model / vega
            x_new = x - step
        bad = ~np.isfinite(x_new) | (x_new <= lo) | (x_new >= hi)
        x_new = np.where(bad, 0.5 * (lo + hi), x_new)
        x = np.where(done, x, x_new)

    s[valid] = x
    return s


def implied_vol(
    price: ArrayLike,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    D: ArrayLike = 1.0,
    is_call: ArrayLike = True,
    **kwargs,
) -> Array:
    """Black-76 implied volatility from a discounted option price."""
    price, F, K, T, D = (np.asarray(x, dtype=float) for x in (price, F, K, T, D))
    k = np.log(K / F)
    s = implied_total_vol(price / (D * F), k, is_call, **kwargs)
    return s / np.sqrt(T)


def implied_vol_bs(
    price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
    is_call: ArrayLike = True,
    **kwargs,
) -> Array:
    """Black-Scholes-Merton implied vol (spot, rate, dividend yield inputs)."""
    S, T, r, q = (np.asarray(x, dtype=float) for x in (S, T, r, q))
    F = S * np.exp((r - q) * T)
    return implied_vol(price, F, K, T, np.exp(-r * T), is_call, **kwargs)
