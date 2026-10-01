"""Analytics on fitted surfaces: term structure, skew, VIX comparison, earnings moves."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)
from scipy.integrate import quad
from scipy.optimize import brentq, nnls

from volsurface.black_scholes import Array, norm_cdf, normalized_price
from volsurface.surface import SVISurface
from volsurface.svi import SVIParams, svi_derivatives, svi_total_variance

TENORS_DAYS = (7, 14, 30, 60, 91, 182, 365)


# ---------------------------------------------------------------------------
# Term structure and skew
# ---------------------------------------------------------------------------


def atm_term_structure(surf: SVISurface, tenors_days=TENORS_DAYS) -> dict[int, float]:
    """ATM (k = 0, i.e. at-the-forward) implied vol at standard tenors.

    Only tenors inside the fitted maturity range are reported; extrapolating
    the surface outside the quoted expiries is not done for analytics.
    """
    T_min, T_max = surf.T[0], surf.T[-1]
    out = {}
    for d in tenors_days:
        T = d / 365
        if T_min <= T <= T_max:
            out[d] = surf.atm_vol(T)
    return out


def delta_strike(surf: SVISurface, T: float, delta: float) -> float:
    """Log-moneyness of the call with forward delta ``delta`` (put: negative).

    Forward (undiscounted) call delta is N(d1), put delta is N(d1) - 1, with
    d1 = -k / sqrt(w(k)) + sqrt(w(k)) / 2 evaluated on the smile.
    """

    def f(k: float) -> float:
        s = float(np.sqrt(surf.total_variance(k, T)[0]))
        d1 = -k / s + s / 2
        n = float(norm_cdf(d1))
        return (n if delta > 0 else n - 1) - delta

    s_atm = float(np.sqrt(surf.total_variance(0.0, T)[0]))
    lo, hi = -12 * s_atm, 12 * s_atm
    return brentq(f, lo, hi, xtol=1e-12)


def skew_metrics(surf: SVISurface, T: float) -> dict[str, float]:
    """Skew measures at maturity T (years):

    * ``rr25``: 25-delta risk reversal, vol(25d call) - vol(25d put); negative
      for equity indices (puts richer than calls).
    * ``bf25``: 25-delta butterfly, average 25d wing vol minus ATM vol.
    * ``skew_90_110``: vol at K = 0.9 F minus vol at K = 1.1 F.
    * ``atm_slope``: d sigma / d k at the money (vol per unit log-moneyness).
    """
    atm = surf.atm_vol(T)
    k_c = delta_strike(surf, T, 0.25)
    k_p = delta_strike(surf, T, -0.25)
    vol_c = float(surf.implied_vol(k_c, T)[0])
    vol_p = float(surf.implied_vol(k_p, T)[0])
    h = 1e-4
    slope = float((surf.implied_vol(h, T)[0] - surf.implied_vol(-h, T)[0]) / (2 * h))
    return {
        "atm": atm,
        "rr25": vol_c - vol_p,
        "bf25": 0.5 * (vol_c + vol_p) - atm,
        "skew_90_110": float(
            surf.implied_vol(np.log(0.9), T)[0] - surf.implied_vol(np.log(1.1), T)[0]
        ),
        "atm_slope": slope,
    }


def svi_atm_slope(p: SVIParams, T: float) -> float:
    """Analytic d sigma / d k at k = 0 for one SVI slice."""
    w, w1, _ = svi_derivatives(p, 0.0)
    return float(w1 / (2 * np.sqrt(w * T)))


# ---------------------------------------------------------------------------
# Model-free implied variance (the VIX construction, on the fitted smile)
# ---------------------------------------------------------------------------


def model_free_variance(p: SVIParams, k_lo: float = -4.0, k_hi: float = 3.0) -> float:
    """Fair total variance of the log-contract from one smile:

        sigma^2 T = 2 * integral q(k) e^{-k} dk

    where q is the undiscounted OTM option price per unit forward. This is the
    continuous-strike limit of the CBOE VIX formula. Integration is truncated
    at ``k_lo`` / ``k_hi``; the tails of an arbitrage-free SVI smile beyond
    that are negligible at the maturities used here.
    """

    def integrand(k: float) -> float:
        s = float(np.sqrt(max(svi_total_variance(p, k), 0.0)))
        return float(normalized_price(k, s, k >= 0)) * np.exp(-k)

    left, _ = quad(integrand, k_lo, 0.0, limit=200)
    right, _ = quad(integrand, 0.0, k_hi, limit=200)
    return 2.0 * (left + right)


def model_free_vol(surf: SVISurface, days: float = 30.0) -> float:
    """Model-free implied vol at ``days``, interpolating the fair total variance
    linearly in maturity between the two bracketing expiries (as the VIX does)."""
    T = days / 365
    Ts = surf.T
    j = int(np.searchsorted(Ts, T))
    if j == 0 or j == len(Ts):
        return float("nan")
    s1, s2 = surf.slices[j - 1], surf.slices[j]
    v1, v2 = model_free_variance(s1.free), model_free_variance(s2.free)
    lam = (T - s1.T) / (s2.T - s1.T)
    return float(np.sqrt(((1 - lam) * v1 + lam * v2) / T))


# ---------------------------------------------------------------------------
# Earnings: implied move from the term structure
# ---------------------------------------------------------------------------


@dataclass
class EarningsMove:
    earnings_date: date
    n_pre: int
    n_post: int
    base_vol: float  # annualised diffusive vol, business-day time
    jump_sd: float  # standard deviation of the earnings-day log return
    expected_abs_move: float  # E|move| = jump_sd * sqrt(2 / pi)
    two_expiry_jump_sd: float  # from the last pre and first post expiry only
    straddle_move: float  # ATM straddle / forward for the first post expiry
    method_note: str


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    rules = [  # noqa: RUF012
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


_HOLIDAYS = NYSEHolidayCalendar().holidays(start="2020-01-01", end="2035-12-31")
NYSE_HOLIDAYS = np.array(_HOLIDAYS.date, dtype="datetime64[D]")


def business_years(trade_date: date, expiry: date) -> float:
    """Trading days from the close of ``trade_date`` to ``expiry``, / 252."""
    return float(np.busday_count(trade_date, expiry, holidays=NYSE_HOLIDAYS)) / 252.0


def earnings_move(
    trade_date: date,
    earnings_date: date,
    expiries: list[date],
    atm_w: Array,
    max_days: int = 120,
) -> EarningsMove | None:
    """Decompose ATM total variance into diffusion plus one earnings jump.

    Model: ``w(T) = sigma_b^2 tau(T) + J^2 1{T after earnings}`` with ``tau`` in
    business-day years (no diffusion over weekends and holidays, so the event
    variance is not smeared over calendar days). Fit by non-negative least
    squares over expiries within ``max_days``. Expiries on the earnings date
    itself are excluded (the timing of the release relative to settlement is
    ambiguous). The straddle estimate ``~ sqrt(2/pi) * total vol`` of the first
    post-earnings expiry includes diffusion and is a known upper bias.
    """
    tau = np.array([business_years(trade_date, e) for e in expiries])
    post = np.array([e > earnings_date for e in expiries])
    keep = np.array(
        [(e - trade_date).days <= max_days and e != earnings_date for e in expiries]
    ) & (tau > 0)
    if keep.sum() < 2 or not post[keep].any():
        return None
    tau_k, post_k, w_k = tau[keep], post[keep], np.asarray(atm_w)[keep]
    A = np.column_stack([tau_k, post_k.astype(float)])
    (s2, j2), _ = nnls(A, w_k)

    pre_idx = np.where(~post_k)[0]
    post_idx = np.where(post_k)[0]
    two = float("nan")
    if len(pre_idx) and len(post_idx):
        i1, i2 = pre_idx[-1], post_idx[0]
        j2_two = w_k[i2] - w_k[i1] / tau_k[i1] * tau_k[i2]
        two = float(np.sqrt(j2_two)) if j2_two > 0 else 0.0
    i_post = post_idx[0]
    straddle = float(2 * normalized_price(0.0, np.sqrt(w_k[i_post]), True))
    return EarningsMove(
        earnings_date=earnings_date,
        n_pre=int((~post_k).sum()),
        n_post=int(post_k.sum()),
        base_vol=float(np.sqrt(s2)),
        jump_sd=float(np.sqrt(j2)),
        expected_abs_move=float(np.sqrt(j2) * np.sqrt(2 / np.pi)),
        two_expiry_jump_sd=two,
        straddle_move=straddle,
        method_note="NNLS on ATM total variance vs business-day time",
    )
