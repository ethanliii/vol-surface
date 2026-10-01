"""Static-arbitrage diagnostics for market quotes and fitted surfaces.

Butterfly (strike) arbitrage
    For one expiry, undiscounted call prices must be decreasing and convex in
    strike with slope >= -1 (per unit forward). On quotes we test this on
    consecutive strikes: a negative butterfly ``C(K1) - (1+lam) C(K2) + lam C(K3)``
    (scaled to unit wing) is an arbitrage. On a fitted smile we use Durrleman's
    condition ``g(k) >= 0``, which is the density being non-negative.

Calendar arbitrage
    With proportional dividends, total implied variance at fixed forward
    log-moneyness must be non-decreasing in maturity: ``w(k, T2) >= w(k, T1)``.

Market quotes are all out-of-the-money quotes that pass the quote-level
filters (bid > 0, not crossed, not very wide, traded in the last 90 days),
*before* any consistency-based stale-quote removal. They are tested two ways:
at mids (noise in the mid alone can produce "violations") and at executable
prices (buy at the ask, sell at the bid), the economically meaningful test.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

from volsurface.black_scholes import Array
from volsurface.ssvi import SSVISurface, ssvi_g
from volsurface.surface import SVISurface
from volsurface.svi import BUTTERFLY_TOL, CALENDAR_TOL, svi_g, svi_total_variance

# Price tolerance for market tests, per unit forward (1e-6 of the forward is
# well below one tick for any underlying here).
PRICE_TOL = 1e-6


def _call_quotes(g: pd.DataFrame) -> pd.DataFrame:
    """Normalized call bid/mid/ask for one expiry from OTM quotes via parity."""
    D, F = g["D"].iloc[0], g["F"].iloc[0]
    out = pd.DataFrame({"x": g["strike"].to_numpy() / F})
    parity = np.where(g["type"].to_numpy() == "P", D * (F - g["strike"].to_numpy()), 0.0)
    for c in ("bid", "mid", "ask"):
        out[c] = (g[c].to_numpy() + parity) / (D * F)
    return out.sort_values("x").drop_duplicates("x").reset_index(drop=True)


def executable_violations(x: Array, bid: Array, ask: Array) -> list[tuple[int, ...]]:
    """Index tuples of quotes forming an executable static arbitrage in one
    expiry. ``x`` sorted strikes / forward; ``bid``/``ask`` normalized call prices.

    * call spread: buy K1 at the ask, sell K2 > K1 at the bid for a credit;
    * put spread (slope < -1): sell K1 at the bid, buy K2 at the ask, credit > K2 - K1;
    * butterfly: buy the wings at the ask, sell the body at the bid for a credit.
    """
    out: list[tuple[int, ...]] = []
    dx = np.diff(x)
    for i in np.where(bid[1:] - ask[:-1] > PRICE_TOL)[0]:
        out.append((int(i), int(i) + 1))
    for i in np.where(bid[:-1] - ask[1:] > dx + PRICE_TOL)[0]:
        out.append((int(i), int(i) + 1))
    if len(x) >= 3:
        lam = (x[2:] - x[1:-1]) / (x[2:] - x[:-2])
        fly = lam * ask[:-2] + (1 - lam) * ask[2:] - bid[1:-1]
        for i in np.where(fly < -PRICE_TOL)[0]:
            out.append((int(i), int(i) + 1, int(i) + 2))
    return out


def market_butterfly(quotes: pd.DataFrame, mask_col: str = "quoted") -> pd.DataFrame:
    """Per expiry: number of strike triples / pairs tested and violations."""
    rows = []
    for expiry, g in quotes[quotes[mask_col]].groupby("expiry"):
        c = _call_quotes(g)
        if len(c) < 3:
            continue
        x, bid, mid, ask = (c[col].to_numpy() for col in ("x", "bid", "mid", "ask"))
        dx = np.diff(x)
        vertical_mid = (np.diff(mid) > PRICE_TOL) | (-np.diff(mid) > dx + PRICE_TOL)
        lam = (x[2:] - x[1:-1]) / (x[2:] - x[:-2])
        fly_mid = lam * mid[:-2] + (1 - lam) * mid[2:] - mid[1:-1]
        ex = executable_violations(x, bid, ask)
        rows.append(
            {
                "expiry": expiry,
                "T": float(g["T"].iloc[0]),
                "n_triples": len(fly_mid),
                "n_pairs": len(dx),
                "fly_viol_mid": int(np.sum(fly_mid < -PRICE_TOL)),
                "fly_viol_exec": sum(len(v) == 3 for v in ex),
                "vertical_viol_mid": int(np.sum(vertical_mid)),
                "vertical_viol_exec": sum(len(v) == 2 for v in ex),
            }
        )
    return pd.DataFrame(rows)


def market_calendar(
    quotes: pd.DataFrame, min_days: float = 0.0, mask_col: str = "quoted"
) -> pd.DataFrame:
    """Consecutive expiries: compare total variance at the longer expiry's
    strikes (within the shorter expiry's quoted range) against the shorter
    expiry's total variance linearly interpolated in k."""
    used = quotes[quotes[mask_col] & (quotes["T"] * 365 >= min_days)]
    groups = [g.sort_values("k") for _, g in used.groupby("expiry", sort=True)]
    rows = []
    for g1, g2 in pairwise(groups):
        k1 = g1["k"].to_numpy()
        T1, T2 = g1["T"].iloc[0], g2["T"].iloc[0]
        inside = g2[(g2["k"] >= k1.min()) & (g2["k"] <= k1.max())]
        if inside.empty:
            continue
        k = inside["k"].to_numpy()
        w1_mid = np.interp(k, k1, g1["w"].to_numpy())
        w1_bid = np.interp(k, k1, g1["iv_bid"].to_numpy() ** 2 * T1)
        w2_mid = inside["w"].to_numpy()
        w2_ask = inside["iv_ask"].to_numpy() ** 2 * T2
        ok = np.isfinite(w1_bid) & np.isfinite(w2_ask)
        rows.append(
            {
                "expiry_1": g1["expiry"].iloc[0],
                "expiry_2": g2["expiry"].iloc[0],
                "n_points": len(k),
                "cal_viol_mid": int(np.sum(w2_mid < w1_mid - CALENDAR_TOL)),
                # Executable: buy the longer expiry at its ask and sell the
                # shorter one at its bid; arbitrage if that is still a credit.
                "cal_viol_exec": int(np.sum(w2_ask[ok] < w1_bid[ok] - CALENDAR_TOL)),
            }
        )
    return pd.DataFrame(rows)


def surface_butterfly(
    params_list: list, k_grid: Array, k_ranges: list[tuple[float, float]], gfun=svi_g
) -> pd.DataFrame:
    rows = []
    for p, (lo, hi) in zip(params_list, k_ranges, strict=True):
        g = gfun(p, k_grid)
        inside = (k_grid >= lo) & (k_grid <= hi)
        rows.append(
            {
                "min_g": float(np.min(g)),
                "viol_points": int(np.sum(g < -BUTTERFLY_TOL)),
                "viol_points_in_data_range": int(np.sum((g < -BUTTERFLY_TOL) & inside)),
            }
        )
    return pd.DataFrame(rows)


def surface_calendar(W: Array, k_grid: Array, k_ranges: list[tuple[float, float]]) -> pd.DataFrame:
    """W: total variance per slice on k_grid, slices sorted by maturity."""
    rows = []
    for i in range(len(W) - 1):
        diff = W[i + 1] - W[i]
        lo = max(k_ranges[i][0], k_ranges[i + 1][0])
        hi = min(k_ranges[i][1], k_ranges[i + 1][1])
        inside = (k_grid >= lo) & (k_grid <= hi)
        rows.append(
            {
                "min_diff": float(np.min(diff)),
                "viol_points": int(np.sum(diff < -CALENDAR_TOL)),
                "viol_points_in_data_range": int(np.sum((diff < -CALENDAR_TOL) & inside)),
            }
        )
    return pd.DataFrame(rows)


def arbitrage_summary(
    quotes: pd.DataFrame, svi: SVISurface, ssvi: SSVISurface | None = None
) -> pd.DataFrame:
    """One row per object: market (mid / executable), raw SVI, arb-free SVI, SSVI.

    Counts are numbers of violating test points: strike triples and pairs of
    consecutive expiries for quotes; grid points (``svi.k_grid``) for fits.
    """
    fit_min_days = min(s.T for s in svi.slices) * 365 - 1e-9
    q = quotes[quotes["T"] * 365 >= fit_min_days]
    fly = market_butterfly(q)
    cal = market_calendar(q)
    grid = svi.k_grid
    ranges = [(s.k.min(), s.k.max()) for s in svi.slices]
    rows = [
        {
            "object": "market quotes (mid)",
            "butterfly_violations": int(fly["fly_viol_mid"].sum() + fly["vertical_viol_mid"].sum()),
            "butterfly_tests": int(fly["n_triples"].sum() + fly["n_pairs"].sum()),
            "calendar_violations": int(cal["cal_viol_mid"].sum()),
            "calendar_tests": int(cal["n_points"].sum()),
            "slices_with_butterfly": int(
                ((fly["fly_viol_mid"] + fly["vertical_viol_mid"]) > 0).sum()
            ),
            "slice_pairs_with_calendar": int((cal["cal_viol_mid"] > 0).sum()),
        },
        {
            "object": "market quotes (executable bid/ask)",
            "butterfly_violations": int(
                fly["fly_viol_exec"].sum() + fly["vertical_viol_exec"].sum()
            ),
            "butterfly_tests": int(fly["n_triples"].sum() + fly["n_pairs"].sum()),
            "calendar_violations": int(cal["cal_viol_exec"].sum()),
            "calendar_tests": int(cal["n_points"].sum()),
            "slices_with_butterfly": int(
                ((fly["fly_viol_exec"] + fly["vertical_viol_exec"]) > 0).sum()
            ),
            "slice_pairs_with_calendar": int((cal["cal_viol_exec"] > 0).sum()),
        },
    ]
    for label, which in (("raw SVI (per slice)", "raw"), ("arbitrage-free SVI", "free")):
        params = svi.params(which)
        bf = surface_butterfly(params, grid, ranges)
        W = np.array([svi_total_variance(p, grid) for p in params])
        cl = surface_calendar(W, grid, ranges)
        rows.append(_fit_row(label, bf, cl, len(grid)))
    if ssvi is not None:
        params = [ssvi.params] * len(ssvi.T)
        bf = pd.DataFrame(
            [
                {
                    "min_g": float(np.min(ssvi_g(ssvi.params, grid, th))),
                    "viol_points": int(np.sum(ssvi_g(ssvi.params, grid, th) < -BUTTERFLY_TOL)),
                }
                for th in ssvi.theta
            ]
        )
        W = np.array([ssvi.total_variance(grid, T) for T in ssvi.T])
        cl = surface_calendar(W, grid, ranges)
        rows.append(_fit_row("SSVI (global)", bf, cl, len(grid)))
    return pd.DataFrame(rows)


def _fit_row(label: str, bf: pd.DataFrame, cl: pd.DataFrame, n_grid: int) -> dict:
    return {
        "object": label,
        "butterfly_violations": int(bf["viol_points"].sum()),
        "butterfly_tests": int(len(bf) * n_grid),
        "calendar_violations": int(cl["viol_points"].sum()),
        "calendar_tests": int(len(cl) * n_grid),
        "slices_with_butterfly": int((bf["viol_points"] > 0).sum()),
        "slice_pairs_with_calendar": int((cl["viol_points"] > 0).sum()),
    }
