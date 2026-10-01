"""End-to-end: synthetic quotes from a known arbitrage-free surface go through
cleaning, IV, SVI fitting and analytics, and the known vols come back out."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from volsurface.black_scholes import black76_price
from volsurface.cleaning import Snapshot, year_fraction
from volsurface.pipeline import process_underlying
from volsurface.rates import RateCurve
from volsurface.ssvi import SSVIParams, ssvi_total_variance

TRADE = date(2026, 10, 1)
TRUE = SSVIParams(rho=-0.7, eta=1.0, gamma=0.4)


def theta(T: float) -> float:
    return 0.02 * T + 0.0002 * np.sqrt(T)


def synthetic_chain(F0: float = 5000.0, r: float = 0.045) -> pd.DataFrame:
    rows = []
    expiries = [date(2026, 10, 9), date(2026, 10, 16), date(2026, 11, 20), date(2026, 12, 18),
                date(2027, 3, 19), date(2027, 9, 17)]  # fmt: skip
    for e in expiries:
        T = year_fraction(TRADE, pd.Series([e]), pd.Series(["SPXW"]))[0]
        F, D = F0 * np.exp(0.01 * T), np.exp(-r * T)
        th = theta(T)
        K = np.arange(np.round(F * np.exp(-5 * np.sqrt(th)), -1), F * np.exp(3 * np.sqrt(th)), 10.0)
        vol = np.sqrt(ssvi_total_variance(TRUE, np.log(K / F), th) / T)
        for kind in ("C", "P"):
            mid = black76_price(F, K, T, vol, D, kind == "C")
            half = np.maximum(0.005 * mid, 0.05)
            ok = mid > 0.2
            rows.append(pd.DataFrame({
                "contractSymbol": [f"SPXW{e:%y%m%d}{kind}{int(k):08d}" for k in K[ok]],
                "strike": K[ok], "bid": mid[ok] - half[ok], "ask": mid[ok] + half[ok],
                "lastPrice": mid[ok], "volume": 5, "openInterest": 50,
                "lastTradeDate": pd.Timestamp("2026-10-01 19:00", tz="UTC"),
                "impliedVolatility": vol[ok], "type": kind, "expiry": e, "root": "SPXW",
            }))  # fmt: skip
    return pd.concat(rows, ignore_index=True)


def test_pipeline_recovers_known_surface() -> None:
    curve = RateCurve(np.array([0.1, 1.0, 2.0]), np.array([0.04, 0.04, 0.04]))
    snap = Snapshot(
        TRADE,
        {"SPX": synthetic_chain()},
        {"underlyings": {"SPX": {"close": 4990.0}}, "vix": {"close": 15.0}},
        curve,
    )
    res = process_underlying(snap, "SPX")
    assert len(res.surface.slices) == 6
    assert res.analytics["median_rmse_vol_free"] < 0.002
    for s in res.surface.slices:
        assert s.info.feasible
        true_atm = np.sqrt(theta(s.T) / s.T)
        assert res.surface.atm_vol(s.T) == pytest.approx(true_atm, abs=0.002)
    # The data come from an arbitrage-free SSVI surface: SSVI fit recovers it.
    assert res.ssvi.params.rho == pytest.approx(TRUE.rho, abs=0.02)
    fitted = res.arbitrage.set_index("object")
    assert fitted.loc["arbitrage-free SVI", "butterfly_violations"] == 0
    assert fitted.loc["arbitrage-free SVI", "calendar_violations"] == 0
    assert fitted.loc["market quotes (executable bid/ask)", "butterfly_violations"] == 0
