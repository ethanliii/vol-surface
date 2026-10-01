from datetime import date

import numpy as np
import pandas as pd
import pytest

from volsurface.black_scholes import black76_price
from volsurface.cleaning import CleanConfig, clean_chain, year_fraction
from volsurface.rates import RateCurve

TRADE_DATE = date(2026, 10, 1)
CURVE = RateCurve(np.array([0.1, 1.0, 5.0]), np.array([0.04, 0.04, 0.04]))


def smile(k: np.ndarray) -> np.ndarray:
    return 0.18 - 0.25 * k + 0.6 * k**2


def synthetic_chain(F: float, r: float, expiry: date, root: str = "SPXW") -> pd.DataFrame:
    T = year_fraction(TRADE_DATE, pd.Series([expiry]), pd.Series([root]))[0]
    D = np.exp(-r * T)
    K = np.arange(np.round(F * 0.7, -1), F * 1.3, 10.0)
    vol = smile(np.log(K / F))
    rows = []
    for kind in ("C", "P"):
        mid = black76_price(F, K, T, vol, D, kind == "C")
        spread = np.maximum(0.02 * mid, 0.10)
        rows.append(
            pd.DataFrame(
                {
                    "contractSymbol": [f"{root}X{kind}{i}" for i in range(len(K))],
                    "strike": K,
                    "bid": np.round(mid - spread / 2, 4),
                    "ask": np.round(mid + spread / 2, 4),
                    "lastPrice": mid,
                    "volume": 10,
                    "openInterest": 100,
                    "lastTradeDate": pd.Timestamp("2026-10-01 19:00", tz="UTC"),
                    "impliedVolatility": vol,
                    "type": kind,
                    "expiry": expiry,
                    "root": root,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_year_fraction_settlement_times() -> None:
    e = pd.Series([date(2026, 10, 16)] * 2)
    pm, am = year_fraction(TRADE_DATE, e, pd.Series(["SPXW", "SPX"]))
    assert pm * 365 == pytest.approx(15.0)
    assert (pm - am) * 365 * 24 == pytest.approx(6.5)


def test_forward_and_discount_recovered_from_parity() -> None:
    F, r = 5000.0, 0.05  # r differs from the curve (4%): must be inferred
    raw = synthetic_chain(F, r, date(2027, 4, 16))
    quotes, fwd = clean_chain(raw, TRADE_DATE, 4900.0, CURVE, CleanConfig())
    assert fwd["F"].iloc[0] == pytest.approx(F, rel=2e-5)
    assert fwd["r_parity"].iloc[0] == pytest.approx(r, abs=2e-3)
    used = quotes[quotes["used"]]
    np.testing.assert_allclose(used["iv"], smile(used["k"]), atol=2e-3)
    # Only OTM options are used.
    assert ((used["k"] >= 0) == (used["type"] == "C")).all()


def test_short_expiry_uses_curve_discount() -> None:
    raw = synthetic_chain(5000.0, 0.04, date(2026, 10, 9))
    _, fwd = clean_chain(raw, TRADE_DATE, 4990.0, CURVE, CleanConfig())
    assert fwd["method"].iloc[0] == "parity_F_curve_D"
    assert fwd["F"].iloc[0] == pytest.approx(5000.0, rel=1e-5)


def test_bad_quotes_are_flagged() -> None:
    raw = synthetic_chain(5000.0, 0.04, date(2027, 1, 15))
    calls = raw.index[raw["type"] == "C"]
    puts = raw.index[raw["type"] == "P"]
    raw.loc[calls[0], "bid"] = 0.0  # zero bid
    raw.loc[calls[1], ["bid", "ask"]] = [10.0, 9.0]  # crossed
    raw.loc[puts[2], ["bid", "ask"]] = [1.0, 5.0]  # very wide
    raw.loc[puts[3], "lastTradeDate"] = pd.Timestamp("2025-06-01", tz="UTC")  # dead line
    # A stale near-the-money call: price 5% too high, breaks parity and the smile.
    atm_call = raw[(raw["type"] == "C") & (raw["strike"] == 5100.0)].index[0]
    raw.loc[atm_call, ["bid", "ask"]] *= 1.08
    # A stale OTM put, IV far off its neighbours.
    otm_put = raw[(raw["type"] == "P") & (raw["strike"] == 4500.0)].index[0]
    raw.loc[otm_put, ["bid", "ask"]] *= 1.6

    quotes, fwd = clean_chain(raw, TRADE_DATE, 4990.0, CURVE, CleanConfig())
    reason = quotes.set_index("contractSymbol")["drop_reason"]
    assert reason[raw.loc[calls[0], "contractSymbol"]] == "zero_bid"
    assert reason[raw.loc[calls[1], "contractSymbol"]] == "crossed_or_locked"
    assert reason[raw.loc[puts[2], "contractSymbol"]] == "wide_market"
    assert reason[raw.loc[puts[3], "contractSymbol"]] == "stale_last_trade"
    assert reason[raw.loc[atm_call, "contractSymbol"]] in {"parity_outlier", "stale_outlier"}
    assert reason[raw.loc[otm_put, "contractSymbol"]] == "stale_outlier"
    assert fwd["F"].iloc[0] == pytest.approx(5000.0, rel=5e-5)


def test_one_root_per_expiry() -> None:
    e = date(2026, 11, 20)
    raw = pd.concat(
        [synthetic_chain(5000.0, 0.04, e, "SPXW"), synthetic_chain(5000.0, 0.04, e, "SPX")]
    )
    raw.loc[raw["root"] == "SPXW", "bid"] = 0.0  # SPXW has no two-sided quotes here
    quotes, _ = clean_chain(raw, TRADE_DATE, 4990.0, CURVE, CleanConfig())
    assert set(quotes["root"]) == {"SPX"}
