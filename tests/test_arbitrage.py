from datetime import date

import numpy as np
import pandas as pd

from volsurface.arbitrage import executable_violations, market_butterfly, market_calendar
from volsurface.black_scholes import normalized_price
from volsurface.cleaning import _arbitrage_stale


def bs_quotes(x: np.ndarray, s: float, half_spread: float = 0.002):
    mid = normalized_price(np.log(x), s, True)
    return mid - half_spread * mid - 1e-4, mid + half_spread * mid + 1e-4


def test_black_scholes_quotes_have_no_executable_arbitrage() -> None:
    x = np.linspace(0.7, 1.3, 61)
    bid, ask = bs_quotes(x, 0.2)
    assert executable_violations(x, bid, ask) == []


def test_detects_each_violation_type() -> None:
    x = np.linspace(0.8, 1.2, 9)
    bid, ask = bs_quotes(x, 0.2)
    b, a = bid.copy(), ask.copy()
    b[5], a[5] = a[4] + 0.01, a[4] + 0.012  # call more expensive than a lower strike
    assert (4, 5) in executable_violations(x, b, a)
    b, a = bid.copy(), ask.copy()
    b[2], a[2] = b[2] + 0.02, a[2] + 0.02  # body above the wings
    viol = executable_violations(x, b, a)
    assert (1, 2, 3) in viol
    b, a = bid.copy(), ask.copy()
    b[0], a[0] = b[1] + (x[1] - x[0]) + 0.01, b[1] + (x[1] - x[0]) + 0.02  # slope < -1
    assert (0, 1) in executable_violations(x, b, a)


def _quotes(strikes, prices, last_trades, expiry=date(2027, 1, 15), T=0.3, F=100.0):
    n = len(strikes)
    return pd.DataFrame(
        {
            "expiry": expiry,
            "T": T,
            "F": F,
            "D": 1.0,
            "strike": strikes,
            "type": "C",
            "bid": np.array(prices) - 0.05,
            "ask": np.array(prices) + 0.05,
            "mid": prices,
            "lastTradeDate": pd.to_datetime(last_trades, utc=True),
            "used": True,
            "quoted": True,
            "k": np.log(np.array(strikes) / F),
        }
    ).assign(contract=[f"c{i}" for i in range(n)])


def test_stale_quote_removed_by_arbitrage_repair() -> None:
    strikes = np.array([100.0, 105, 110, 115, 120])
    prices = normalized_price(np.log(strikes / 100), 0.15, True) * 100
    prices[2] += 2.0  # stale body: executable butterfly and call-spread arbitrage
    last = ["2026-10-01"] * 5
    last[2] = "2026-08-01"
    g = _quotes(strikes, prices, last)
    dropped = _arbitrage_stale(g)
    assert list(g.loc[dropped, "contract"]) == ["c2"]
    assert market_butterfly(g.drop(dropped))["fly_viol_exec"].sum() == 0
    assert market_butterfly(g)["fly_viol_exec"].sum() >= 1


def test_market_calendar_detects_crossing() -> None:
    k = np.linspace(-0.2, 0.2, 9)
    rows = []
    for expiry, T, vol in ((date(2026, 11, 20), 0.14, 0.20), (date(2026, 12, 18), 0.22, 0.15)):
        iv = np.full_like(k, vol)
        rows.append(
            pd.DataFrame(
                {
                    "expiry": expiry,
                    "T": T,
                    "k": k,
                    "iv": iv,
                    "w": iv**2 * T,
                    "iv_bid": iv - 0.002,
                    "iv_ask": iv + 0.002,
                    "quoted": True,
                }
            )
        )
    q = pd.concat(rows)
    # 0.15^2 * 0.22 = 0.00495 < 0.20^2 * 0.14 = 0.0056: calendar arbitrage at every strike.
    cal = market_calendar(q)
    assert cal["cal_viol_mid"].iloc[0] == 9
    assert cal["cal_viol_exec"].iloc[0] == 9
