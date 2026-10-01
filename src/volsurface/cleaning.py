"""Quote cleaning, forward / discount-factor inference, and implied vols.

Pipeline for one underlying on one day:

1. Time to expiry in ACT/365 calendar time from the 16:00 ET close to the
   settlement time (09:30 ET for AM-settled SPX monthlies, 16:00 ET otherwise).
2. On days where an expiry has both SPX (AM) and SPXW (PM) series, keep the
   root with more two-sided quotes so each expiry has one settlement time.
3. Quote filters, each recorded as a ``drop_reason``: zero bid, crossed or
   locked market, very wide market (relative spread above a threshold), and
   stale lines that have not traded for months.
4. Per expiry, infer forward ``F`` and discount factor ``D`` from put-call
   parity ``C - P = D (F - K)`` by weighted least squares on near-the-money
   strikes, iteratively discarding pairs inconsistent with parity (stale lines).
5. Keep only out-of-the-money options relative to ``F`` (puts below, calls at
   or above), which removes deep in-the-money quotes. Compute mid, bid and ask
   implied vols.
6. Flag remaining stale quotes as implied-vol outliers versus their strike
   neighbours. Yahoo provides no quote timestamps, so staleness can only be
   detected from cross-sectional inconsistency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from volsurface.iv import implied_total_vol
from volsurface.rates import RateCurve

NY = ZoneInfo("America/New_York")
YEAR_SECONDS = 365.0 * 24 * 3600
AM_SETTLED_ROOTS = {"SPX"}


@dataclass(frozen=True)
class CleanConfig:
    european: bool = True
    max_rel_spread: float = 0.5  # (ask - bid) / mid
    max_trade_age_days: int = 90  # no trade for this long -> quote treated as stale
    min_days: float = 1.0  # drop expiries closer than this (calendar days)
    parity_window_sd: float = 1.0  # parity strikes within this many ATM st.devs
    parity_min_window: float = 0.02  # ... but at least this |log-moneyness|
    parity_max_pairs: int = 30
    min_days_implied_rate: float = 30.0  # below this, D comes from the rate curve
    max_rate_deviation: float = 0.02  # |r_parity - r_curve| tolerated (cc, decimal)
    outlier_window: int = 5  # strikes in the rolling median
    outlier_min_vol: float = 0.02  # vol points tolerated beyond the bid-ask band
    outlier_spread_mult: float = 3.0
    min_quotes_per_expiry: int = 8


SPX_CONFIG = CleanConfig()
STOCK_CONFIG = CleanConfig(european=False, max_rel_spread=0.5)


@dataclass
class Snapshot:
    trade_date: date
    chains: dict[str, pd.DataFrame]
    meta: dict
    curve: RateCurve


def load_snapshot(snap_dir: Path) -> Snapshot:
    meta = json.loads((snap_dir / "meta.json").read_text())
    chains = {name: pd.read_parquet(snap_dir / f"{name}.parquet") for name in meta["underlyings"]}
    curve = RateCurve.from_frame(pd.read_parquet(snap_dir / "rates.parquet"))
    return Snapshot(date.fromisoformat(meta["trade_date"]), chains, meta, curve)


def list_snapshots(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.iterdir() if (p / "meta.json").exists())


def year_fraction(trade_date: date, expiry: pd.Series, root: pd.Series) -> np.ndarray:
    """ACT/365 from the 16:00 ET close to settlement (AM settlement for SPX roots)."""
    start = datetime.combine(trade_date, time(16, 0), NY)
    out = np.empty(len(expiry))
    for i, (e, r) in enumerate(zip(expiry, root, strict=True)):
        settle = time(9, 30) if r in AM_SETTLED_ROOTS else time(16, 0)
        out[i] = (datetime.combine(e, settle, NY) - start).total_seconds() / YEAR_SECONDS
    return out


def _choose_root(df: pd.DataFrame) -> pd.DataFrame:
    """One option root per expiry (the one with more two-sided quotes)."""
    two_sided = (df["bid"] > 0) & (df["ask"] > df["bid"])
    counts = df[two_sided].groupby(["expiry", "root"], observed=True).size().rename("n")
    best = counts.reset_index().sort_values("n").groupby("expiry").tail(1)
    keep = set(zip(best["expiry"], best["root"], strict=True))
    mask = [(e, r) in keep for e, r in zip(df["expiry"], df["root"], strict=True)]
    return df[mask]


def apply_quote_filters(df: pd.DataFrame, trade_date: date, cfg: CleanConfig) -> pd.DataFrame:
    df = df.copy()
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["drop_reason"] = ""

    def flag(mask, reason):
        df.loc[mask & (df["drop_reason"] == ""), "drop_reason"] = reason

    flag(~(df["bid"] > 0), "zero_bid")
    flag(~(df["ask"] > df["bid"]), "crossed_or_locked")
    flag((df["ask"] - df["bid"]) / df["mid"] > cfg.max_rel_spread, "wide_market")
    # Yahoo has no quote timestamps. A line that has not traded for months can
    # carry a frozen quote (e.g. NFLX pre-split strikes in 2026 still showing
    # 2025 markets), so a long gap since the last trade marks the quote stale.
    last_trade = pd.to_datetime(df["lastTradeDate"], utc=True).dt.date
    age = np.array([(trade_date - d).days if pd.notna(d) else 10**6 for d in last_trade])
    flag(age > cfg.max_trade_age_days, "stale_last_trade")
    return df


@dataclass
class ForwardFit:
    expiry: date
    T: float
    F: float
    D: float
    r_curve: float
    r_parity: float  # NaN if D was not estimated from parity
    method: str
    n_pairs: int
    n_parity_outliers: int
    rmse: float  # parity residual RMSE, in price units


def infer_forward(
    pairs: pd.DataFrame, T: float, spot: float, curve: RateCurve, cfg: CleanConfig
) -> tuple[ForwardFit, pd.Index]:
    """Fit C - P = D F - D K on near-the-money pairs.

    ``pairs`` has columns strike, c_mid, p_mid, c_spread, p_spread. Returns the
    fit and the index of pairs rejected as parity outliers.
    """
    r_curve = float(curve.rate(T))
    D_curve = float(np.exp(-r_curve * T))
    y = (pairs["c_mid"] - pairs["p_mid"]).to_numpy()
    K = pairs["strike"].to_numpy()
    sigma_y = np.sqrt(pairs["c_spread"] ** 2 + pairs["p_spread"] ** 2).to_numpy() / 2

    # Initial forward: where |C - P| is smallest, corrected by parity with D_curve.
    i0 = int(np.argmin(np.abs(y)))
    F = K[i0] + y[i0] / D_curve
    # Window: rough ATM st.dev from the straddle at that strike: straddle ~ 0.8 F s.
    straddle = pairs["c_mid"].iloc[i0] + pairs["p_mid"].iloc[i0]
    s_atm = straddle / (0.8 * F * D_curve)
    window = max(cfg.parity_min_window, cfg.parity_window_sd * s_atm)

    use_parity_D = cfg.european and cfg.min_days_implied_rate <= T * 365
    # Robust pre-screen: per-strike implied forwards K + (C - P) / D_curve should
    # agree; reject strikes far from the median (stale lines) before any least
    # squares, so a few bad quotes cannot drag the regression.
    F_k = K + y / D_curve
    order = np.argsort(np.abs(np.log(K / F)))
    core = order[: cfg.parity_max_pairs]
    med = np.median(F_k[core])
    mad = 1.4826 * np.median(np.abs(F_k[core] - med))
    keep = np.ones(len(K), dtype=bool)  # strikes outside the core are never judged
    keep[core] = np.abs(F_k[core] - med) <= np.maximum(5 * mad, 3 * sigma_y[core] / D_curve) + 0.05
    F = med
    D = D_curve
    for _ in range(5):
        near = np.abs(np.log(K / F)) <= window
        # Limit to the closest pairs to the forward.
        order = np.argsort(np.abs(np.log(K / F)))
        nearest = np.zeros_like(near)
        nearest[order[: cfg.parity_max_pairs]] = True
        sel = keep & near & nearest
        if sel.sum() < 2:
            sel = keep & nearest
        w = 1.0 / np.maximum(sigma_y[sel], 1e-6) ** 2
        if use_parity_D and sel.sum() >= 5:
            A = np.column_stack([np.ones(sel.sum()), -K[sel]])
            Aw = A * np.sqrt(w)[:, None]
            coef, *_ = np.linalg.lstsq(Aw, y[sel] * np.sqrt(w), rcond=None)
            D_hat = coef[1]
            r_hat = -np.log(D_hat) / T if D_hat > 0 else np.nan
            if np.isfinite(r_hat) and abs(r_hat - r_curve) <= cfg.max_rate_deviation:
                D, F = D_hat, coef[0] / D_hat
                method = "parity_F_and_D"
            else:
                D = D_curve
                F = np.average(K[sel] + y[sel] / D, weights=w)
                method = "parity_F_curve_D (implied rate rejected)"
        else:
            D = D_curve
            F = np.average(K[sel] + y[sel] / D, weights=w)
            method = "parity_F_curve_D"
        resid = y - D * (F - K)
        # Parity outliers: residual beyond 3x the quoted uncertainty (+ 1 tick floor).
        tol = 3.0 * sigma_y + 0.05
        new_keep = keep & ~(near & nearest & (np.abs(resid) > tol))
        if new_keep.sum() == keep.sum() or new_keep[sel].sum() < 2:
            break
        keep = new_keep
    sel_final = keep & near & nearest
    rmse = float(np.sqrt(np.mean(resid[sel_final] ** 2))) if sel_final.any() else np.nan
    fit = ForwardFit(
        expiry=pairs["expiry"].iloc[0],
        T=T,
        F=float(F),
        D=float(D),
        r_curve=r_curve,
        r_parity=float(-np.log(D) / T) if method == "parity_F_and_D" else np.nan,
        method=method,
        n_pairs=int(sel_final.sum()),
        n_parity_outliers=int((~keep).sum()),
        rmse=rmse,
    )
    return fit, pairs.index[~keep]


def _flag_iv_outliers(g: pd.DataFrame, cfg: CleanConfig) -> pd.Series:
    """True for quotes whose IV is far from the rolling median of neighbours."""
    g = g.sort_values("k")
    med = g["iv"].rolling(cfg.outlier_window, center=True, min_periods=3).median()
    half_spread = (g["iv_ask"] - g["iv_bid"]).abs() / 2
    tol = np.maximum(cfg.outlier_spread_mult * half_spread.fillna(0), cfg.outlier_min_vol)
    return ((g["iv"] - med).abs() > tol).reindex(g.index)


def clean_chain(
    raw: pd.DataFrame,
    trade_date: date,
    spot: float,
    curve: RateCurve,
    cfg: CleanConfig = SPX_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (quotes, forwards).

    ``quotes`` contains every raw quote with ``drop_reason`` ("" if used) and,
    for OTM quotes, ``k``, ``T``, ``iv``, ``iv_bid``, ``iv_ask``, ``w``.
    ``forwards`` has one row per expiry with F, D and diagnostics.
    """
    df = raw.copy()
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    df = _choose_root(df)
    df["T"] = year_fraction(trade_date, df["expiry"], df["root"].astype(str))
    df = df[df["T"] * 365 >= cfg.min_days]
    df = apply_quote_filters(df, trade_date, cfg)

    fwd_rows: list[ForwardFit] = []
    out = []
    for expiry, g in df.groupby("expiry", sort=True):
        T = float(g["T"].iloc[0])
        good = g[g["drop_reason"] == ""]
        calls = good[good["type"] == "C"].set_index("strike")
        puts = good[good["type"] == "P"].set_index("strike")
        common = calls.index.intersection(puts.index)
        if len(common) < 3:
            g = g.assign(drop_reason=g["drop_reason"].replace("", "no_forward"))
            out.append(g)
            continue
        pairs = pd.DataFrame(
            {
                "expiry": expiry,
                "strike": common,
                "c_mid": calls.loc[common, "mid"].to_numpy(),
                "p_mid": puts.loc[common, "mid"].to_numpy(),
                "c_spread": (calls.loc[common, "ask"] - calls.loc[common, "bid"]).to_numpy(),
                "p_spread": (puts.loc[common, "ask"] - puts.loc[common, "bid"]).to_numpy(),
            }
        ).reset_index(drop=True)
        fit, bad_pairs = infer_forward(pairs, T, spot, curve, cfg)
        fwd_rows.append(fit)

        g = g.copy()
        g["F"], g["D"] = fit.F, fit.D
        g["k"] = np.log(g["strike"] / fit.F)
        bad_strikes = set(pairs.loc[bad_pairs, "strike"])
        g.loc[g["strike"].isin(bad_strikes) & (g["drop_reason"] == ""), "drop_reason"] = (
            "parity_outlier"
        )
        otm = np.where(g["k"] >= 0, g["type"] == "C", g["type"] == "P")
        g.loc[~otm & (g["drop_reason"] == ""), "drop_reason"] = "in_the_money"

        norm = g["D"] * g["F"]
        is_call = (g["type"] == "C").to_numpy()
        k = g["k"].to_numpy()
        sqrt_t = np.sqrt(T)
        with np.errstate(invalid="ignore", divide="ignore"):
            g["iv"] = implied_total_vol(g["mid"] / norm, k, is_call) / sqrt_t
            g["iv_bid"] = implied_total_vol(g["bid"] / norm, k, is_call) / sqrt_t
            g["iv_ask"] = implied_total_vol(g["ask"] / norm, k, is_call) / sqrt_t
        g.loc[g["iv"].isna() & (g["drop_reason"] == ""), "drop_reason"] = "no_implied_vol"

        live = g["drop_reason"] == ""
        if live.sum() >= 3:
            outl = _flag_iv_outliers(g[live], cfg)
            g.loc[outl[outl].index, "drop_reason"] = "stale_outlier"
        if (g["drop_reason"] == "").sum() < cfg.min_quotes_per_expiry:
            g.loc[g["drop_reason"] == "", "drop_reason"] = "too_few_quotes"
        out.append(g)

    quotes = pd.concat(out, ignore_index=True)
    quotes["w"] = quotes["iv"] ** 2 * quotes["T"]
    quotes["used"] = quotes["drop_reason"] == ""
    forwards = pd.DataFrame([f.__dict__ for f in fwd_rows])
    if not forwards.empty:
        forwards["spot"] = spot
        forwards["carry"] = np.log(forwards["F"] / spot) / forwards["T"]  # r - q implied
    return quotes, forwards


def cleaning_report(quotes: pd.DataFrame) -> pd.Series:
    reasons = quotes["drop_reason"].replace("", "used")
    return reasons.value_counts()
