"""Figures: static PNGs (matplotlib) for the README and interactive plotly figures.

Colour roles follow one fixed categorical order so an entity keeps its colour
across every chart: market data in ink/grey, arbitrage-free SVI in blue, raw
SVI in orange, SSVI in aqua.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from volsurface.analysis import business_years
from volsurface.surface import SVISurface
from volsurface.svi import svi_g, svi_total_variance

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SPREAD = "#c3c2b7"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
    }
)


def _pct(ax, axis="y"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _title(fig, title: str, subtitle: str | None = None) -> float:
    """Title + subtitle placed in inches from the top; returns the top of the
    plotting area as a figure fraction."""
    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.12 / h, title, ha="left", va="top", fontsize=12, weight="bold", color=INK)
    if subtitle:
        fig.text(0.01, 1 - 0.40 / h, subtitle, ha="left", va="top", fontsize=9, color=INK2)
    return 1 - 0.75 / h


def pick_slices(surf: SVISurface, targets_days=(7, 14, 30, 60, 91, 182, 365, 730), n=6):
    """Indices of slices nearest to target maturities (unique, sorted)."""
    days = surf.T * 365
    idx = sorted({int(np.argmin(np.abs(days - d))) for d in targets_days if d <= days[-1] * 1.1})
    if len(idx) > n:
        idx = [idx[round(i)] for i in np.linspace(0, len(idx) - 1, n)]
    return idx


# ---------------------------------------------------------------------------
# Static figures (PNG)
# ---------------------------------------------------------------------------


def smile_grid(surf: SVISurface, ssvi, name: str, date_str: str, path: Path) -> None:
    """Smiles with bid/ask, fits and residuals for a set of maturities."""
    idx = pick_slices(surf)
    ncol = 3
    nrow = int(np.ceil(len(idx) / ncol))
    fig = plt.figure(figsize=(11, 4.0 * nrow + 0.6))
    ratios = ([3, 1.1, 0.75] * nrow)[:-1]
    gs = fig.add_gridspec(3 * nrow - 1, ncol, height_ratios=ratios, hspace=0.12, wspace=0.22)
    for n, i in enumerate(idx):
        s = surf.slices[i]
        r, c = divmod(n, ncol)
        ax = fig.add_subplot(gs[3 * r, c])
        axr = fig.add_subplot(gs[3 * r + 1, c], sharex=ax)
        lo, hi = s.k.min(), s.k.max()
        pad = 0.08 * (hi - lo)
        kk = np.linspace(lo - pad, hi + pad, 300)
        T = s.T
        ax.vlines(s.k, s.iv_bid, s.iv_ask, color=SPREAD, lw=1.2, label="bid–ask")
        ax.plot(s.k, s.iv, "o", ms=2.2, color=INK, label="mid", zorder=3)
        ax.plot(
            kk, np.sqrt(svi_total_variance(s.raw, kk) / T), color=ORANGE, lw=1.4, label="raw SVI"
        )
        if ssvi is not None:
            ax.plot(kk, ssvi.implied_vol(kk, T), color=AQUA, lw=1.4, label="SSVI")
        ax.plot(
            kk,
            np.sqrt(svi_total_variance(s.free, kk) / T),
            color=BLUE,
            lw=2,
            label="arbitrage-free SVI",
        )
        ax.set_title(f"{s.expiry}  ({T * 365:.0f}d)", loc="left")
        _pct(ax)
        plt.setp(ax.get_xticklabels(), visible=False)
        # Residuals (vol points) with the half-spread band.
        fit = np.sqrt(svi_total_variance(s.free, s.k) / T)
        half = (s.iv_ask - s.iv_bid) / 2
        axr.fill_between(
            s.k,
            -half * 100,
            half * 100,
            color=SPREAD,
            alpha=0.45,
            lw=0,
            step="mid",
            label="±half-spread",
        )
        axr.axhline(0, color=AXIS, lw=0.8)
        axr.plot(s.k, (s.iv - fit) * 100, "o", ms=2, color=BLUE)
        lim = max(0.3, float(np.nanpercentile(np.abs(s.iv - fit) * 100, 98)) * 1.3)
        axr.set_ylim(-lim, lim)
        axr.set_xlabel("log-moneyness k = ln(K/F)")
        if c == 0:
            ax.set_ylabel("implied vol")
            axr.set_ylabel("mid − fit\n(vol pts)")
        if n == 0:
            ax.legend(loc="upper right", handlelength=1.2)
    _ = _title(
        fig,
        f"{name} implied-vol smiles, {date_str}",
        "Market bid–ask (grey bars), mids (black), fitted smiles; lower panels: residual "
        "of the arbitrage-free fit vs the half-spread band",
    )
    fig.subplots_adjust(top=1 - 1.0 / fig.get_figheight(), left=0.07, right=0.98, bottom=0.07)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def term_structure(surfaces: dict[str, SVISurface], date_str: str, path: Path) -> None:
    """ATM vol vs maturity: SPX and the single stocks as small multiples."""
    names = list(surfaces)
    ncol = min(len(names), 3)
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.2 * nrow + 0.6), squeeze=False)
    for ax, name in zip(axes.flat, names, strict=False):
        surf = surfaces[name]
        days = surf.T * 365
        mkt = np.array([np.sqrt(s.theta_market / s.T) for s in surf.slices])
        fit = np.array([surf.atm_vol(T) for T in surf.T])
        ax.plot(days, fit, color=BLUE, label="arbitrage-free SVI")
        ax.plot(days, mkt, "o", ms=4, color=INK, mec=SURFACE, mew=1, label="market (interp. at F)")
        ax.set_xscale("log")
        ax.set_title(name, loc="left")
        ax.set_xlabel("days to expiry (log scale)")
        _pct(ax)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    for ax in axes.flat[len(names) :]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="lower right")
    axes.flat[0].set_ylabel("ATM implied vol")
    top = _title(
        fig,
        f"ATM implied-vol term structure, {date_str}",
        "At-the-forward vol by expiry. Single stocks show the earnings bump: vol "
        "jumps for the first expiry after the report.",
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def skew_term_structure(surf: SVISurface, date_str: str, path: Path, name: str = "SPX") -> None:
    from volsurface.analysis import skew_metrics

    rows = [skew_metrics(surf, T) | {"days": T * 365} for T in surf.T]
    df = pd.DataFrame(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    a1.plot(df["days"], df["rr25"] * 100, color=BLUE)
    a1.set_title("25-delta risk reversal (vol pts)", loc="left")
    a1.set_ylabel("σ(25Δ call) − σ(25Δ put)")
    a2.plot(df["days"], df["atm_slope"], color=BLUE)
    a2.set_title("ATM skew dσ/dk", loc="left")
    a2.set_ylabel("vol per unit log-moneyness")
    for a in (a1, a2):
        a.set_xscale("log")
        a.set_xlabel("days to expiry (log scale)")
        a.axhline(0, color=AXIS, lw=0.8)
        a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    top = _title(
        fig,
        f"{name} skew term structure, {date_str}",
        "Puts are richer than calls at every maturity. The ATM slope dσ/dk flattens "
        "with maturity while the 25Δ risk reversal widens in vol points.",
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def history_chart(hist: pd.DataFrame, path: Path) -> None:
    """SPX 30d ATM vol, model-free 30d vol and VIX (and skew once there is history)."""
    h = hist[hist["name"] == "SPX"].sort_values("date")
    series = [
        ("vix_close", "VIX close", INK, 0.01),
        ("model_free_vol_30d", "model-free 30d (SVI)", BLUE, 1.0),
        ("atm_30d", "ATM 30d (SVI)", ORANGE, 1.0),
    ]
    n = len(h)
    if n == 1:
        fig, a1 = plt.subplots(figsize=(8, 3.0))
        row = h.iloc[0]
        vals = [row[c] * sc for c, _, _, sc in series]
        y = np.arange(len(vals))[::-1]
        a1.scatter(
            vals,
            y,
            s=70,
            c=[c for _, _, c, _ in series],
            edgecolors=SURFACE,
            linewidths=2,
            zorder=3,
        )
        for yi, v in zip(y, vals, strict=True):
            a1.text(v + 0.0012, yi, f"{v * 100:.2f}%", va="center", color=INK, fontsize=9)
        a1.set_yticks(y, [lab for _, lab, _, _ in series])
        a1.set_ylim(-0.6, len(vals) - 0.4)
        a1.grid(axis="y", visible=False)
        _pct(a1, "x")
        a1.set_xlim(min(vals) - 0.01, max(vals) + 0.01)
        a1.set_title(f"30-day implied vol on {h['date'].iloc[0]:%Y-%m-%d}", loc="left")
        note = "Only one snapshot so far; this becomes a time series as daily snapshots accumulate."
    else:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
        for col, lab, colr, scale in series:
            a1.plot(h["date"], h[col] * scale, color=colr, marker="o", ms=3, label=lab)
        _pct(a1)
        a1.legend()
        a1.set_title("30-day implied vol", loc="left")
        for col, lab, colr in (("rr25_30d", "30d", BLUE), ("rr25_91d", "91d", ORANGE)):
            a2.plot(h["date"], h[col] * 100, color=colr, marker="o", ms=3, label=lab)
        a2.legend()
        a2.set_title("25Δ risk reversal (vol pts)", loc="left")
        fig.autofmt_xdate()
        note = f"{n} daily snapshots, {h['date'].min():%Y-%m-%d} to {h['date'].max():%Y-%m-%d}."
    top = _title(fig, "SPX: fitted surface vs VIX", note)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def earnings_chart(surfaces: dict, analytics: dict, trade_date, path: Path) -> None:
    """ATM total variance vs business time for each stock with the fitted
    diffusion + jump decomposition."""
    names = [n for n in surfaces if (analytics[n].get("earnings") or None)]
    if not names:
        return
    ncol = 2
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.3 * nrow + 0.7), squeeze=False)
    for ax, name in zip(axes.flat, names, strict=False):
        surf, e = surfaces[name], analytics[name]["earnings"]
        edate = pd.Timestamp(e["earnings_date"]).date()
        tau = np.array([business_years(trade_date, s.expiry) for s in surf.slices]) * 252
        w = np.array([s.theta_market for s in surf.slices])
        post = np.array([s.expiry > edate for s in surf.slices])
        keep = tau <= 120 * 252 / 365
        ax.plot(
            tau[keep & ~post],
            w[keep & ~post] * 1e4,
            "o",
            color=INK,
            ms=5,
            mec=SURFACE,
            label="expiry before earnings",
        )
        ax.plot(
            tau[keep & post],
            w[keep & post] * 1e4,
            "o",
            color=BLUE,
            ms=5,
            mec=SURFACE,
            label="expiry after earnings",
        )
        t = np.linspace(0, tau[keep].max() * 1.02, 200)
        te = business_years(trade_date, edate) * 252
        model = (e["base_vol"] ** 2 * t / 252 + e["jump_sd"] ** 2 * (t > te)) * 1e4
        ax.plot(t, model, color=BLUE, lw=1.5, label="fit: σ²τ + J²·1{after}")
        ax.axvline(te, color=AXIS, lw=1)
        ax.text(
            te,
            ax.get_ylim()[1] * 0.97,
            f" earnings {edate:%b %d}",
            color=INK2,
            va="top",
            fontsize=8,
        )
        ax.set_title(
            f"{name}: implied earnings move ±{e['jump_sd'] * 100:.1f}% "
            f"(1 sd), E|move| {e['expected_abs_move'] * 100:.1f}%",
            loc="left",
        )
        ax.set_xlabel("trading days to expiry")
        ax.set_ylabel("ATM total variance (×10⁻⁴)")
    for ax in axes.flat[len(names) :]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="lower right")
    top = _title(
        fig,
        f"Earnings moves implied by the term structure, {trade_date}",
        "Market ATM total variance per expiry vs trading-day time: the jump at the "
        "earnings date is the variance of the earnings-day move.",
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def arbitrage_chart(surf: SVISurface, name: str, date_str: str, path: Path) -> None:
    """Durrleman g for the raw fit with the worst violation, and the repaired fit."""
    worst = int(np.argmin([svi_g(s.raw, surf.k_grid).min() for s in surf.slices]))
    s = surf.slices[worst]
    lo, hi = min(s.k.min(), -0.5), max(s.k.max(), 0.5)
    k = np.linspace(lo - 0.5, hi + 0.8, 600)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    a1.axvspan(s.k.min(), s.k.max(), color=GRID, alpha=0.5, lw=0, label="quoted strikes")
    a1.axhline(0, color=AXIS, lw=0.8)
    a1.plot(k, svi_g(s.raw, k), color=ORANGE, label="raw SVI")
    a1.plot(k, svi_g(s.free, k), color=BLUE, label="arbitrage-free SVI")
    a1.set_title(f"Durrleman g(k), {s.expiry} ({s.T * 365:.0f}d)", loc="left")
    a1.set_xlabel("log-moneyness k")
    a1.set_ylabel("g(k)   (density ≥ 0 iff g ≥ 0)")
    a1.legend(loc="upper left")
    a2.axvspan(s.k.min(), s.k.max(), color=GRID, alpha=0.5, lw=0)
    a2.plot(s.k, s.iv, "o", ms=2.2, color=INK, label="market mid")
    a2.plot(
        k, np.sqrt(np.maximum(svi_total_variance(s.raw, k), 0) / s.T), color=ORANGE, label="raw SVI"
    )
    a2.plot(k, np.sqrt(svi_total_variance(s.free, k) / s.T), color=BLUE, label="arbitrage-free SVI")
    _pct(a2)
    a2.set_title("Same slice: implied vol", loc="left")
    a2.set_xlabel("log-moneyness k")
    a2.legend(loc="upper right")
    top = _title(
        fig,
        f"{name}: butterfly arbitrage in an unconstrained fit, {date_str}",
        "The raw fit matches the quotes but implies a negative density in the "
        "extrapolated wing; the constrained fit removes it.",
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Interactive figures (plotly)
# ---------------------------------------------------------------------------


def surface_figure(surf: SVISurface, quotes: pd.DataFrame, name: str, date_str: str) -> go.Figure:
    """3D implied-vol surface over (moneyness K/F, days) with market mids."""
    T_grid = np.geomspace(surf.T[0], surf.T[-1], 120)
    lo = max(-0.6, min(s.k.min() for s in surf.slices))
    hi = min(0.3, max(s.k.max() for s in surf.slices))
    k = np.linspace(lo, hi, 150)
    Z = np.array([surf.implied_vol(k, T) for T in T_grid])
    # Draw the surface over a smooth region in standardized moneyness
    # z = k / sqrt(w_atm(T)), from 4 ATM standard deviations below the forward
    # to 2 above, so short maturities are not stretched over far wings.
    sd = np.sqrt([surf.total_variance(0.0, T)[0] for T in T_grid])
    Z[(k[None, :] < -4 * sd[:, None]) | (k[None, :] > 2 * sd[:, None])] = np.nan
    X = np.exp(k)
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=X,
            y=T_grid * 365,
            z=Z * 100,
            colorscale=[[i / 6, c] for i, c in enumerate(BLUE_RAMP)],
            colorbar={"title": "vol %", "len": 0.6, "thickness": 12},
            opacity=0.95,
            name="arbitrage-free SVI",
            hovertemplate="K/F %{x:.3f}<br>%{y:.0f} days<br>vol %{z:.2f}%<extra>SVI</extra>",
            contours={"z": {"show": False}},
        )
    )
    used = quotes[quotes["used"] & quotes["k"].between(lo, hi) & (quotes["T"] >= surf.T[0])]
    sd_q = np.sqrt([surf.total_variance(0.0, T)[0] for T in used["T"]])
    used = used[(used["k"] >= -4 * sd_q) & (used["k"] <= 2 * sd_q)]
    # Thin to ~25 quotes per expiry so the surface stays visible.
    used = used.groupby("expiry", group_keys=False).apply(lambda g: g.iloc[:: max(1, len(g) // 25)])
    fig.add_trace(
        go.Scatter3d(
            x=np.exp(used["k"]),
            y=used["T"] * 365,
            z=used["iv"] * 100,
            mode="markers",
            marker={"size": 2, "color": ORANGE},
            name="market mids (subsample)",
            hovertemplate="K/F %{x:.3f}<br>%{y:.1f} days<br>mid vol %{z:.2f}%<extra>market</extra>",
        )
    )
    fig.update_layout(
        title={"text": f"{name} implied volatility surface, {date_str}", "x": 0.01},
        scene={
            "xaxis_title": "moneyness K/F",
            "xaxis": {"range": [float(np.exp(lo)), float(np.exp(hi))]},
            "yaxis_title": "days to expiry",
            "zaxis_title": "implied vol (%)",
            "yaxis": {"type": "log", "tickvals": [2, 7, 30, 90, 180, 365, 730]},
            "camera": {
                "eye": {"x": 1.45, "y": -1.3, "z": 0.6},
                "center": {"x": 0, "y": 0, "z": -0.12},
            },
            "aspectmode": "manual",
            "aspectratio": {"x": 1.2, "y": 1.4, "z": 0.7},
        },
        legend={"x": 0.01, "y": 0.95},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        paper_bgcolor=SURFACE,
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif", "color": INK},
    )
    return _f32(fig)


def _f32(fig: go.Figure) -> go.Figure:
    """Store trace arrays as float32 to halve the embedded JSON size."""
    for t in fig.data:
        for attr in ("x", "y", "z"):
            v = getattr(t, attr, None)
            if v is not None and hasattr(v, "dtype") and v.dtype.kind == "f":
                setattr(t, attr, np.asarray(v, dtype=np.float32))
    return fig


def smile_explorer(surf: SVISurface, ssvi, name: str) -> go.Figure:
    """One smile at a time with a dropdown over expiries."""
    fig = go.Figure()
    per = 5
    for i, s in enumerate(surf.slices):
        vis = i == 0
        pad = 0.08 * (s.k.max() - s.k.min())
        kk = np.linspace(s.k.min() - pad, s.k.max() + pad, 120)
        T = s.T
        fig.add_trace(
            go.Scatter(
                x=np.repeat(s.k, 3),
                y=np.column_stack([s.iv_bid, s.iv_ask, np.full_like(s.k, np.nan)]).ravel() * 100,
                mode="lines",
                line={"color": SPREAD, "width": 1.5},
                name="bid–ask",
                hoverinfo="skip",
                visible=vis,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=s.k,
                y=s.iv * 100,
                mode="markers",
                name="mid",
                marker={"size": 4, "color": INK},
                visible=vis,
                hovertemplate="k %{x:.3f}<br>mid %{y:.2f}%<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=kk,
                y=np.sqrt(svi_total_variance(s.raw, kk) / T) * 100,
                name="raw SVI",
                line={"color": ORANGE, "width": 1.5},
                visible=vis,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=kk,
                y=ssvi.implied_vol(kk, T) * 100,
                name="SSVI",
                line={"color": AQUA, "width": 1.5},
                visible=vis,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=kk,
                y=np.sqrt(svi_total_variance(s.free, kk) / T) * 100,
                name="arbitrage-free SVI",
                line={"color": BLUE, "width": 2.5},
                visible=vis,
            )
        )
    buttons = []
    for i, s in enumerate(surf.slices):
        vis = [False] * (per * len(surf.slices))
        vis[per * i : per * (i + 1)] = [True] * per
        buttons.append(
            {
                "label": f"{s.expiry} ({s.T * 365:.0f}d)",
                "method": "update",
                "args": [{"visible": vis}],
            }
        )
    fig.update_layout(
        updatemenus=[
            {"buttons": buttons, "x": 0.01, "y": 1.13, "xanchor": "left", "showactive": True}
        ],
        title={"text": f"{name} smile by expiry", "x": 0.01, "y": 0.97},
        xaxis_title="log-moneyness k = ln(K/F)",
        yaxis_title="implied vol (%)",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        hovermode="closest",
        margin={"l": 60, "r": 20, "t": 90, "b": 50},
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif", "color": INK},
        xaxis={"gridcolor": GRID, "zeroline": False},
        yaxis={"gridcolor": GRID},
        legend={"orientation": "h", "y": -0.18},
    )
    return _f32(fig)
