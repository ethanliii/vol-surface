"""Daily option-chain snapshots from Yahoo Finance (via yfinance).

Layout on disk (one directory per trading day)::

    data/snapshots/2026-10-01/
        SPX.parquet  JPM.parquet ...   # one row per option quote
        rates.parquet                  # Treasury curve used that day
        meta.json                      # spot, VIX, earnings dates, fetch time

Only raw quotes are stored; all cleaning happens downstream so that cleaning
rules can be changed and the full history re-processed.

Caveats about the source (documented in the README):
* Yahoo does not publish a quote timestamp, only the time of the last trade.
  Snapshots are taken after the close, so bid/ask are the closing quotes.
* SPX index options are European and cash-settled; single-stock options are
  American, which biases Black-Scholes implied vols slightly upward for
  in-the-money puts (we only use out-of-the-money quotes for that reason).
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from volsurface.rates import fetch_curve

log = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")
DATA_DIR = Path("data/snapshots")

INDEX_TICKERS = ["^SPX"]
# Single stocks chosen for earnings in October 2026 (dates from Yahoo, stored per snapshot).
EARNINGS_TICKERS = ["JPM", "NFLX", "TSLA", "META"]
MAX_EXPIRY_YEARS = {"^SPX": 3.0}
DEFAULT_MAX_EXPIRY_YEARS = 1.0

KEEP_COLS = [
    "contractSymbol",
    "strike",
    "bid",
    "ask",
    "lastPrice",
    "volume",
    "openInterest",
    "lastTradeDate",
    "impliedVolatility",
]


def clean_name(ticker: str) -> str:
    return ticker.lstrip("^")


def option_root(contract_symbol: str) -> str:
    """'SPXW261016C05000000' -> 'SPXW'."""
    m = re.match(r"^([A-Z]+?)\d{6}[CP]\d{8}$", contract_symbol)
    return m.group(1) if m else ""


def last_session_date(ticker: str = "^SPX") -> date:
    hist = yf.Ticker(ticker).history(period="10d")
    return pd.Timestamp(hist.index[-1]).date()


def _retry(fn, tries: int = 4, wait: float = 3.0):
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:
            if i == tries - 1:
                raise
            log.warning("retrying after %s", exc)
            time.sleep(wait * (i + 1))
    return None


def fetch_chain(ticker: str, trade_date: date, pause: float = 0.3) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    expiries = _retry(lambda: tk.options)
    max_years = MAX_EXPIRY_YEARS.get(ticker, DEFAULT_MAX_EXPIRY_YEARS)
    frames = []
    for exp in expiries:
        exp_date = date.fromisoformat(exp)
        if exp_date <= trade_date or (exp_date - trade_date).days > 365 * max_years:
            continue
        chain = _retry(lambda e=exp: tk.option_chain(e))
        for kind, df in (("C", chain.calls), ("P", chain.puts)):
            if df.empty:
                continue
            df = df[[c for c in KEEP_COLS if c in df.columns]].copy()
            df["type"] = kind
            df["expiry"] = exp_date
            frames.append(df)
        time.sleep(pause)
    out = pd.concat(frames, ignore_index=True)
    out["root"] = out["contractSymbol"].map(option_root)
    out["lastTradeDate"] = pd.to_datetime(out["lastTradeDate"], utc=True)
    for c in ("volume", "openInterest"):
        out[c] = out[c].fillna(0).astype("int64")
    out["type"] = out["type"].astype("category")
    out["root"] = out["root"].astype("category")
    return out


def underlying_info(ticker: str) -> dict:
    tk = yf.Ticker(ticker)
    hist = _retry(lambda: tk.history(period="5d"))
    info: dict = {
        "close": float(hist["Close"].iloc[-1]),
        "close_date": str(pd.Timestamp(hist.index[-1]).date()),
    }
    if not ticker.startswith("^"):
        try:
            cal = tk.calendar or {}
            dates = cal.get("Earnings Date") or []
            info["earnings_dates"] = [str(d) for d in dates]
        except Exception as exc:  # calendar endpoint is flaky; not fatal
            log.warning("no earnings calendar for %s: %s", ticker, exc)
    return info


def collect_snapshot(
    out_dir: Path = DATA_DIR,
    tickers: list[str] | None = None,
    force: bool = False,
) -> Path | None:
    """Fetch one snapshot. Returns the snapshot directory, or None if skipped."""
    tickers = tickers or INDEX_TICKERS + EARNINGS_TICKERS
    now_ny = datetime.now(NY)
    trade_date = last_session_date()
    if trade_date != now_ny.date() and not force:
        log.info("no session today (%s, last %s); skipping", now_ny.date(), trade_date)
        return None
    snap_dir = out_dir / trade_date.isoformat()
    snap_dir.mkdir(parents=True, exist_ok=True)

    meta: dict = {
        "trade_date": trade_date.isoformat(),
        "fetched_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "yahoo finance via yfinance " + yf.__version__,
        "underlyings": {},
    }
    for ticker in tickers:
        name = clean_name(ticker)
        try:
            chain = fetch_chain(ticker, trade_date)
            chain.to_parquet(snap_dir / f"{name}.parquet", compression="zstd", index=False)
            info = underlying_info(ticker)
            info["n_quotes"] = len(chain)
            info["n_expiries"] = int(chain["expiry"].nunique())
            meta["underlyings"][name] = info
            log.info("%s: %d quotes, %d expiries", name, len(chain), info["n_expiries"])
        except Exception as exc:
            log.error("failed to collect %s: %s", ticker, exc)

    vix = yf.Ticker("^VIX").history(period="5d")
    meta["vix"] = {
        "close": float(vix["Close"].iloc[-1]),
        "date": str(pd.Timestamp(vix.index[-1]).date()),
    }
    curve = fetch_curve(trade_date)
    curve.to_frame().to_parquet(snap_dir / "rates.parquet", index=False)
    meta["rates"] = {"source": curve.source, "asof": str(curve.asof)}
    (snap_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return snap_dir
