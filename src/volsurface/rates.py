"""Risk-free rate curve from US Treasury par yields (FRED as a fallback).

The Treasury publishes daily constant-maturity par yields on a bond-equivalent
(semi-annual) basis. We convert each to a continuously compounded rate,
``r_c = 2 * ln(1 + y / 2)``, and interpolate linearly in maturity. Treating par
yields as zero rates is an approximation, but at option maturities (< 2y) the
difference is a few basis points, far below the precision with which we can
infer discount factors from option prices. For SPX the curve is only a prior
and a sanity check: the per-expiry discount factor is inferred from put-call
parity (see ``volsurface.cleaning``).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import requests

TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

TREASURY_TENORS: dict[str, float] = {
    "1 Mo": 1 / 12,
    "1.5 Month": 1.5 / 12,
    "2 Mo": 2 / 12,
    "3 Mo": 3 / 12,
    "4 Mo": 4 / 12,
    "6 Mo": 6 / 12,
    "1 Yr": 1.0,
    "2 Yr": 2.0,
    "3 Yr": 3.0,
    "5 Yr": 5.0,
    "7 Yr": 7.0,
    "10 Yr": 10.0,
}
FRED_SERIES: dict[str, float] = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 3 / 12,
    "DGS6MO": 6 / 12,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS5": 5.0,
    "DGS10": 10.0,
}
HEADERS = {"User-Agent": "Mozilla/5.0 (volsurface research project)"}


@dataclass(frozen=True)
class RateCurve:
    """Continuously compounded zero-rate curve, linear in maturity, flat beyond the ends."""

    tenors: np.ndarray  # years, increasing
    rates: np.ndarray  # continuously compounded
    asof: date | None = None
    source: str = ""

    def rate(self, t: np.ndarray | float) -> np.ndarray:
        return np.interp(np.asarray(t, dtype=float), self.tenors, self.rates)

    def discount(self, t: np.ndarray | float) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.exp(-self.rate(t) * t)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"tenor": self.tenors, "rate_cc": self.rates, "asof": self.asof, "source": self.source}
        )

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> RateCurve:
        df = df.sort_values("tenor")
        asof = pd.Timestamp(df["asof"].iloc[0]).date() if "asof" in df else None
        source = str(df["source"].iloc[0]) if "source" in df else ""
        return cls(df["tenor"].to_numpy(float), df["rate_cc"].to_numpy(float), asof, source)


def bey_to_continuous(y_pct: np.ndarray) -> np.ndarray:
    """Bond-equivalent (semi-annual) yield in percent -> continuously compounded decimal."""
    return 2.0 * np.log1p(np.asarray(y_pct, dtype=float) / 200.0)


def _curve_from_row(row: pd.Series, tenor_map: dict[str, float], asof: date, source: str):
    pairs = [(t, float(row[c])) for c, t in tenor_map.items() if c in row and pd.notna(row[c])]
    if len(pairs) < 3:
        raise ValueError(f"too few tenors in {source} curve for {asof}")
    tenors, ylds = map(np.array, zip(*sorted(pairs), strict=True))
    return RateCurve(tenors, bey_to_continuous(ylds), asof, source)


def fetch_treasury_curve(asof: date, timeout: float = 30) -> RateCurve:
    """Treasury par curve for the latest date on or before ``asof``."""
    frames = []
    for year in sorted({asof.year, asof.year - 1}):
        resp = requests.get(TREASURY_URL.format(year=year), headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        frames.append(pd.read_csv(io.StringIO(resp.text)))
    df = pd.concat(frames)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y").dt.date
    df = df[df["Date"] <= asof].sort_values("Date")
    if df.empty:
        raise ValueError(f"no Treasury curve on or before {asof}")
    row = df.iloc[-1]
    return _curve_from_row(row, TREASURY_TENORS, row["Date"], "treasury")


def fetch_fred_curve(asof: date, timeout: float = 30) -> RateCurve:
    """Same curve built from FRED constant-maturity series (fallback)."""
    cols = {}
    for series in FRED_SERIES:
        resp = requests.get(FRED_URL.format(series=series), headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        s = pd.read_csv(io.StringIO(resp.text), index_col=0, na_values=".").iloc[:, 0]
        s.index = pd.to_datetime(s.index).date
        cols[series] = s
    df = pd.DataFrame(cols)
    df = df[df.index <= asof].dropna(how="all")
    row = df.iloc[-1]
    return _curve_from_row(row, FRED_SERIES, row.name, "fred")


def fetch_curve(asof: date) -> RateCurve:
    try:
        return fetch_treasury_curve(asof)
    except Exception:
        return fetch_fred_curve(asof)
