"""End-to-end processing of a snapshot, with derived results cached on disk.

For each snapshot date and underlying, ``process_snapshot`` cleans quotes,
fits the SVI and SSVI surfaces, runs the arbitrage diagnostics and computes
analytics. Results are written to ``data/derived/<date>/`` so the history can
be rebuilt incrementally as snapshots accumulate; bumping ``PIPELINE_VERSION``
invalidates the cache when the methodology changes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from volsurface.analysis import (
    atm_term_structure,
    earnings_move,
    model_free_vol,
    skew_metrics,
    svi_atm_slope,
)
from volsurface.arbitrage import arbitrage_summary
from volsurface.cleaning import (
    SPX_CONFIG,
    STOCK_CONFIG,
    Snapshot,
    clean_chain,
    cleaning_report,
    list_snapshots,
    load_snapshot,
)
from volsurface.ssvi import SSVISurface, fit_ssvi
from volsurface.surface import SVISurface, build_slices, fit_svi_surface
from volsurface.svi import SVIParams

log = logging.getLogger(__name__)

PIPELINE_VERSION = "2"
DERIVED_DIR = Path("data/derived")
INDEXES = {"SPX"}


@dataclass
class Result:
    name: str
    trade_date: date
    quotes: pd.DataFrame
    forwards: pd.DataFrame
    surface: SVISurface
    ssvi: SSVISurface
    fit_summary: pd.DataFrame
    arbitrage: pd.DataFrame
    cleaning: pd.Series
    analytics: dict


def process_underlying(snap: Snapshot, name: str) -> Result:
    european = name in INDEXES
    cfg = SPX_CONFIG if european else STOCK_CONFIG
    info = snap.meta["underlyings"][name]
    quotes, forwards = clean_chain(
        snap.chains[name], snap.trade_date, info["close"], snap.curve, cfg
    )
    surf = fit_svi_surface(build_slices(quotes, min_days=2.0))
    sl = surf.slices
    ssvi = fit_ssvi(
        [s.k for s in sl],
        [s.w for s in sl],
        [s.weights for s in sl],
        surf.T,
        np.array([s.theta_market for s in sl]),
    )
    summary = surf.summary()
    summary["rmse_vol_ssvi"] = [
        float(np.sqrt(np.mean((ssvi.implied_vol(s.k, s.T) - s.iv) ** 2))) for s in sl
    ]
    summary["atm_slope"] = [svi_atm_slope(s.free, s.T) for s in sl]
    arb = arbitrage_summary(quotes, surf, ssvi)

    analytics: dict = {
        "name": name,
        "trade_date": snap.trade_date.isoformat(),
        "spot": info["close"],
        "european": european,
        "atm": {str(d): v for d, v in atm_term_structure(surf).items()},
        "n_expiries_fitted": len(sl),
        "n_quotes_used": int(quotes["used"].sum()),
        "median_rmse_vol_free": float(summary["rmse_vol_free"].median()),
        "median_rmse_vol_raw": float(summary["rmse_vol_raw"].median()),
        "median_rmse_vol_ssvi": float(summary["rmse_vol_ssvi"].median()),
        "ssvi": asdict(ssvi.params),
    }
    for days in (30, 91):
        T = days / 365
        if surf.T[0] <= T <= surf.T[-1]:
            analytics[f"skew_{days}d"] = skew_metrics(surf, T)
    if european:
        analytics["model_free_vol_30d"] = model_free_vol(surf, 30)
        vix = snap.meta.get("vix", {})
        analytics["vix_close"] = vix.get("close")
    dates = info.get("earnings_dates") or []
    if dates:
        e_date = date.fromisoformat(dates[0])
        # Market ATM total variance (mid IVs interpolated at k = 0): model-free,
        # so an SVI misfit at the money (e.g. TSLA's sharp ATM curvature) does
        # not leak into the jump estimate.
        atm_w = np.array([s.theta_market for s in sl])
        mv = earnings_move(snap.trade_date, e_date, [s.expiry for s in sl], atm_w)
        analytics["earnings"] = (
            None
            if mv is None
            else {
                **{k: v for k, v in asdict(mv).items() if k != "earnings_date"},
                "earnings_date": e_date.isoformat(),
            }
        )
    return Result(
        name=name,
        trade_date=snap.trade_date,
        quotes=quotes,
        forwards=forwards,
        surface=surf,
        ssvi=ssvi,
        fit_summary=summary,
        arbitrage=arb,
        cleaning=cleaning_report(quotes),
        analytics=analytics,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

QUOTE_COLS = [
    "expiry", "T", "root", "type", "strike", "bid", "ask", "mid", "F", "D", "k",
    "iv", "iv_bid", "iv_ask", "w", "used", "quoted", "drop_reason",
]  # fmt: skip


def save_result(res: Result, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = res.name
    q = res.quotes[res.quotes["quoted"]][QUOTE_COLS].copy()
    q.to_parquet(out_dir / f"{n}_quotes.parquet", compression="zstd", index=False)
    res.forwards.to_parquet(out_dir / f"{n}_forwards.parquet", index=False)
    fs = res.fit_summary.copy()
    for i, s in enumerate(res.surface.slices):
        for key, val in s.raw.as_dict().items():
            fs.loc[i, f"raw_{key}"] = val
    fs.to_parquet(out_dir / f"{n}_svi.parquet", index=False)
    res.arbitrage.to_parquet(out_dir / f"{n}_arbitrage.parquet", index=False)
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "analytics": res.analytics,
        "cleaning": {str(k): int(v) for k, v in res.cleaning.items()},
        "ssvi": {"T": res.ssvi.T.tolist(), "theta": res.ssvi.theta.tolist()},
        "k_grid": [
            float(res.surface.k_grid[0]),
            float(res.surface.k_grid[-1]),
            len(res.surface.k_grid),
        ],
    }
    (out_dir / f"{n}_analytics.json").write_text(json.dumps(payload, indent=1, default=_json))


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, date):
        return o.isoformat()
    raise TypeError(type(o))


def is_cached(out_dir: Path, name: str) -> bool:
    f = out_dir / f"{name}_analytics.json"
    if not f.exists():
        return False
    return json.loads(f.read_text()).get("pipeline_version") == PIPELINE_VERSION


def process_snapshot(snap_dir: Path, derived_root: Path = DERIVED_DIR, force: bool = False):
    snap = load_snapshot(snap_dir)
    out_dir = derived_root / snap.trade_date.isoformat()
    results = {}
    for name in snap.meta["underlyings"]:
        if not force and is_cached(out_dir, name):
            continue
        try:
            res = process_underlying(snap, name)
        except Exception:
            log.exception("failed to process %s %s", snap.trade_date, name)
            continue
        save_result(res, out_dir)
        results[name] = res
        log.info("%s %s: %d slices", snap.trade_date, name, len(res.surface.slices))
    return results


def process_all(data_dir: Path, derived_root: Path = DERIVED_DIR, force: bool = False) -> None:
    for snap_dir in list_snapshots(data_dir):
        process_snapshot(snap_dir, derived_root, force)


# ---------------------------------------------------------------------------
# Loading derived results
# ---------------------------------------------------------------------------


def clean_quotes(snap_dir: Path, name: str) -> pd.DataFrame:
    snap = load_snapshot(snap_dir)
    cfg = SPX_CONFIG if name in INDEXES else STOCK_CONFIG
    close = snap.meta["underlyings"][name]["close"]
    quotes, _ = clean_chain(snap.chains[name], snap.trade_date, close, snap.curve, cfg)
    return quotes[quotes["quoted"]][QUOTE_COLS]


def load_surface(
    derived_dir: Path, name: str, snapshots_dir: Path = Path("data/snapshots")
) -> tuple[SVISurface, pd.DataFrame]:
    """Rebuild an SVISurface (params only) and the quote table from disk."""
    from volsurface.surface import Slice

    fs = pd.read_parquet(derived_dir / f"{name}_svi.parquet")
    qfile = derived_dir / f"{name}_quotes.parquet"
    if qfile.exists():
        quotes = pd.read_parquet(qfile)
    else:  # not committed to git: re-clean from the raw snapshot (fast, deterministic)
        quotes = clean_quotes(snapshots_dir / derived_dir.name, name)
    meta = json.loads((derived_dir / f"{name}_analytics.json").read_text())
    used = quotes[quotes["used"]]
    slices = []
    for _, row in fs.iterrows():
        g = used[used["expiry"] == row["expiry"]].sort_values("k")
        slices.append(
            Slice(
                expiry=row["expiry"],
                T=float(row["days"]) / 365,
                F=float(g["F"].iloc[0]) if len(g) else np.nan,
                k=g["k"].to_numpy(float),
                w=g["w"].to_numpy(float),
                iv=g["iv"].to_numpy(float),
                iv_bid=g["iv_bid"].to_numpy(float),
                iv_ask=g["iv_ask"].to_numpy(float),
                weights=np.ones(len(g)),
                raw=SVIParams(*(float(row[f"raw_{c}"]) for c in ("a", "b", "rho", "m", "sigma"))),
                free=SVIParams(*(float(row[c]) for c in ("a", "b", "rho", "m", "sigma"))),
            )
        )
    lo, hi, n = meta["k_grid"]
    return SVISurface(slices, np.linspace(lo, hi, int(n))), quotes


def load_history(derived_root: Path = DERIVED_DIR) -> pd.DataFrame:
    """One row per (date, underlying) with the scalar analytics."""
    rows = []
    for f in sorted(derived_root.glob("*/*_analytics.json")):
        a = json.loads(f.read_text())["analytics"]
        row = {"date": pd.Timestamp(a["trade_date"]), "name": a["name"], "spot": a["spot"]}
        for d, v in a.get("atm", {}).items():
            row[f"atm_{d}d"] = v
        for key in ("skew_30d", "skew_91d"):
            for m, v in (a.get(key) or {}).items():
                row[f"{m}_{key[5:]}"] = v
        for key in ("model_free_vol_30d", "vix_close", "median_rmse_vol_free"):
            if key in a:
                row[key] = a[key]
        e = a.get("earnings")
        if e:
            row["earnings_date"] = e["earnings_date"]
            row["earnings_jump_sd"] = e["jump_sd"]
            row["earnings_expected_abs_move"] = e["expected_abs_move"]
        rows.append(row)
    return pd.DataFrame(rows)
