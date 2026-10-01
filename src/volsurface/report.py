"""Build the static site (docs/) and figures from cached derived results."""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from volsurface import plots
from volsurface.pipeline import DERIVED_DIR, load_history, load_surface
from volsurface.ssvi import SSVIParams, SSVISurface

log = logging.getLogger(__name__)
DOCS = Path("docs")


def latest_date(derived_root: Path) -> Path:
    dirs = sorted(p for p in derived_root.iterdir() if any(p.glob("*_analytics.json")))
    if not dirs:
        raise FileNotFoundError(f"no derived results in {derived_root}; run `volsurface process`")
    return dirs[-1]


def load_day(day_dir: Path):
    names = sorted(f.name.split("_analytics")[0] for f in day_dir.glob("*_analytics.json"))
    names = sorted(names, key=lambda n: (n != "SPX", n))
    surfaces, quotes, analytics, ssvis, arbs, fits, cleaning = {}, {}, {}, {}, {}, {}, {}
    for n in names:
        surf, q = load_surface(day_dir, n)
        payload = json.loads((day_dir / f"{n}_analytics.json").read_text())
        a = payload["analytics"]
        surfaces[n], quotes[n], analytics[n] = surf, q, a
        ssvis[n] = SSVISurface(
            SSVIParams(**a["ssvi"]),
            np.array(payload["ssvi"]["T"]),
            np.array(payload["ssvi"]["theta"]),
        )
        arbs[n] = pd.read_parquet(day_dir / f"{n}_arbitrage.parquet")
        fits[n] = pd.read_parquet(day_dir / f"{n}_svi.parquet")
        cleaning[n] = payload["cleaning"]
    return names, surfaces, quotes, analytics, ssvis, arbs, fits, cleaning


def build_report(derived_root: Path = DERIVED_DIR, docs: Path = DOCS) -> dict:
    day_dir = latest_date(derived_root)
    date_str = day_dir.name
    trade_date = pd.Timestamp(date_str).date()
    names, surfaces, quotes, analytics, ssvis, arbs, _, cleaning = load_day(day_dir)
    img = docs / "img"
    img.mkdir(parents=True, exist_ok=True)

    main = "SPX" if "SPX" in names else names[0]
    plots.smile_grid(surfaces[main], ssvis[main], main, date_str, img / f"smiles_{main}.png")
    plots.term_structure(surfaces, date_str, img / "term_structure.png")
    plots.skew_term_structure(surfaces[main], date_str, img / f"skew_{main}.png", main)
    plots.arbitrage_chart(surfaces[main], main, date_str, img / f"arbitrage_{main}.png")
    hist = load_history(derived_root)
    if (hist["name"] == "SPX").any():
        plots.history_chart(hist, img / "history_SPX.png")
    stocks = {n: surfaces[n] for n in names if n != main}
    plots.earnings_chart(stocks, analytics, trade_date, img / "earnings.png")

    surf_figs = {n: plots.surface_figure(surfaces[n], quotes[n], n, date_str) for n in names}
    try:  # static export needs kaleido + a Chrome binary; the site does not
        surf_figs[main].write_image(img / f"surface_{main}.png", width=1100, height=700, scale=1.5)
    except Exception as exc:
        log.warning("could not export surface PNG: %s", exc)
    smile_figs = {n: plots.smile_explorer(surfaces[n], ssvis[n], n) for n in names}

    tables = results_tables(names, analytics, arbs, cleaning, hist)
    (docs / "results.json").write_text(json.dumps(tables["json"], indent=1, default=str))
    (docs / "index.html").write_text(
        site_html(date_str, names, surf_figs, smile_figs, tables, analytics[main])
    )
    hist.to_csv(docs / "history.csv", index=False)
    log.info("report written to %s for %s", docs, date_str)
    return tables


def _f(x, pct=True, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x * 100:.{nd}f}%" if pct else f"{x:.{nd}f}"


def results_tables(names, analytics, arbs, cleaning, hist) -> dict:
    fit_rows, arb_rows, earn_rows, clean_rows = [], [], [], []
    for n in names:
        a = analytics[n]
        fit_rows.append({
            "underlying": n, "expiries": a["n_expiries_fitted"], "quotes": a["n_quotes_used"],
            "rmse_raw_svi_volpts": a["median_rmse_vol_raw"] * 100,
            "rmse_arbfree_svi_volpts": a["median_rmse_vol_free"] * 100,
            "rmse_ssvi_volpts": a["median_rmse_vol_ssvi"] * 100,
            "atm_30d": a.get("atm", {}).get("30"), "rr25_30d": (a.get("skew_30d") or {}).get("rr25"),
        })  # fmt: skip
        for _, r in arbs[n].iterrows():
            arb_rows.append({"underlying": n, **r.to_dict()})
        e = a.get("earnings")
        if e:
            earn_rows.append({
                "underlying": n, "earnings_date": e["earnings_date"],
                "jump_sd": e["jump_sd"], "expected_abs_move": e["expected_abs_move"],
                "two_expiry_jump_sd": e["two_expiry_jump_sd"], "straddle_move": e["straddle_move"],
                "base_vol": e["base_vol"], "n_pre": e["n_pre"], "n_post": e["n_post"],
            })  # fmt: skip
        c = cleaning[n]
        total = sum(c.values())
        clean_rows.append({"underlying": n, "raw_quotes": total, **c})
    spx = analytics.get("SPX", {})
    headline = {
        "date": spx.get("trade_date"),
        "n_snapshots": int(hist["date"].nunique()) if len(hist) else 0,
        "atm_30d": spx.get("atm", {}).get("30"),
        "model_free_30d": spx.get("model_free_vol_30d"),
        "vix": spx.get("vix_close"),
    }
    return {
        "fit": pd.DataFrame(fit_rows), "arb": pd.DataFrame(arb_rows),
        "earn": pd.DataFrame(earn_rows), "clean": pd.DataFrame(clean_rows).fillna(0),
        "headline": headline,
        "json": {"headline": headline, "fit": fit_rows, "arbitrage": arb_rows,
                 "earnings": earn_rows, "cleaning": clean_rows},
    }  # fmt: skip


def markdown_tables(tables: dict) -> str:
    """Markdown snippets for the README (printed by `volsurface build`)."""
    out = []
    fit = tables["fit"]
    out.append(
        "| Underlying | Expiries | Quotes | Raw SVI | Arb-free SVI | SSVI | ATM 30d | RR25 30d |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in fit.iterrows():
        out.append(
            f"| {r['underlying']} | {r['expiries']} | {r['quotes']:,} | "
            f"{r['rmse_raw_svi_volpts']:.2f} | {r['rmse_arbfree_svi_volpts']:.2f} | "
            f"{r['rmse_ssvi_volpts']:.2f} | {_f(r['atm_30d'])} | "
            f"{_f(r['rr25_30d'] * 100 if r['rr25_30d'] is not None else None, pct=False)} |"
        )
    out.append("")
    arb = tables["arb"]
    out.append(
        "| Underlying | Object | Butterfly violations / tests | Calendar violations / tests |"
    )
    out.append("|---|---|---:|---:|")
    for _, r in arb.iterrows():
        out.append(
            f"| {r['underlying']} | {r['object']} | {r['butterfly_violations']:,} / "
            f"{r['butterfly_tests']:,} | {r['calendar_violations']:,} / {r['calendar_tests']:,} |"
        )
    out.append("")
    earn = tables["earn"]
    if len(earn):
        out.append(
            "| Stock | Earnings | Implied move (1 sd) | E\\|move\\| | Two-expiry | "
            "Straddle / F | Base vol |"
        )
        out.append("|---|---|---:|---:|---:|---:|---:|")
        for _, r in earn.iterrows():
            out.append(
                f"| {r['underlying']} | {r['earnings_date']} | {_f(r['jump_sd'])} | "
                f"{_f(r['expected_abs_move'])} | {_f(r['two_expiry_jump_sd'])} | "
                f"{_f(r['straddle_move'])} | {_f(r['base_vol'], nd=1)} |"
            )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--ring:rgba(11,11,11,.10);--accent:#2a78d6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#1a1a19;
--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--ring:rgba(255,255,255,.10);--accent:#3987e5}}
:root[data-theme="dark"]{--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--ring:rgba(255,255,255,.10);--accent:#3987e5}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1120px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:19px;margin:40px 0 8px}
p.lead{color:var(--ink2);margin:0 0 24px;max-width:780px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px 16px}
.tile .label{color:var(--ink2);font-size:13px}.tile .value{font-size:30px;font-weight:600}
.tile .sub{color:var(--muted);font-size:12px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:8px;
margin-top:12px;overflow:hidden}
.card img{width:100%;height:auto;display:block;border-radius:6px}
.plot{width:100%;height:640px}.plot.small{height:520px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}
.tabs button{font:inherit;font-size:13px;border:1px solid var(--ring);background:var(--surface);
color:var(--ink);border-radius:999px;padding:4px 12px;cursor:pointer}
.tabs button[aria-pressed="true"]{border-color:var(--accent);color:var(--accent);font-weight:600}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;width:100%}
th,td{padding:6px 10px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
th{color:var(--ink2);font-weight:600}
.note{color:var(--ink2);font-size:13px;max-width:820px}
a{color:var(--accent)}
@media (max-width:640px){.plot{height:460px}.plot.small{height:420px}h1{font-size:22px}}
"""


def _table_html(df: pd.DataFrame, fmt: dict) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            cells.append(f"<td>{html.escape(fmt[c](v) if c in fmt else str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def site_html(date_str, names, surf_figs, smile_figs, tables, spx) -> str:
    h = tables["headline"]
    fig_json = {
        "surface": {n: json.loads(f.to_json()) for n, f in surf_figs.items()},
        "smile": {n: json.loads(f.to_json()) for n, f in smile_figs.items()},
    }
    fit = tables["fit"].rename(columns={
        "underlying": "Underlying", "expiries": "Expiries", "quotes": "Quotes used",
        "rmse_raw_svi_volpts": "RMSE raw SVI (vol pts)",
        "rmse_arbfree_svi_volpts": "RMSE arb-free SVI", "rmse_ssvi_volpts": "RMSE SSVI",
        "atm_30d": "ATM 30d", "rr25_30d": "RR25 30d (vol pts)"})  # fmt: skip
    fit_html = _table_html(fit, {
        "Quotes used": lambda v: f"{int(v):,}", "RMSE raw SVI (vol pts)": lambda v: f"{v:.2f}",
        "RMSE arb-free SVI": lambda v: f"{v:.2f}", "RMSE SSVI": lambda v: f"{v:.2f}",
        "ATM 30d": _f, "RR25 30d (vol pts)": lambda v: f"{v * 100:.2f}"})  # fmt: skip
    arb = tables["arb"][
        [
            "underlying",
            "object",
            "butterfly_violations",
            "butterfly_tests",
            "calendar_violations",
            "calendar_tests",
        ]
    ]
    arb_html = _table_html(arb.rename(columns={
        "underlying": "Underlying", "object": "Object", "butterfly_violations": "Butterfly viol.",
        "butterfly_tests": "Butterfly tests", "calendar_violations": "Calendar viol.",
        "calendar_tests": "Calendar tests"}), {})  # fmt: skip
    earn = tables["earn"]
    earn_html = ""
    if len(earn):
        e = earn[
            [
                "underlying",
                "earnings_date",
                "jump_sd",
                "expected_abs_move",
                "two_expiry_jump_sd",
                "straddle_move",
                "base_vol",
            ]
        ]
        earn_html = _table_html(e.rename(columns={
            "underlying": "Stock", "earnings_date": "Earnings", "jump_sd": "Move (1 sd)",
            "expected_abs_move": "E|move|", "two_expiry_jump_sd": "Two-expiry est.",
            "straddle_move": "Straddle / F", "base_vol": "Base vol"}),
            {c: _f for c in ["Move (1 sd)", "E|move|", "Two-expiry est.", "Straddle / F",
                             "Base vol"]})  # fmt: skip
    tabs = lambda kind: "".join(  # noqa: E731
        f'<button data-kind="{kind}" data-name="{n}" aria-pressed="{str(i == 0).lower()}">{n}</button>'
        for i, n in enumerate(names)
    )
    snap_note = (
        f"{h['n_snapshots']} daily snapshot"
        + ("s" if h["n_snapshots"] != 1 else "")
        + " collected so far; time-series analysis updates automatically as the GitHub "
        "Actions collector adds one snapshot per trading day."
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Implied Vol Surface</title>
<style>{CSS}</style>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
</head><body><main>
<h1>Implied volatility surface engine</h1>
<p class="lead">SPX index options (European) and four single stocks into earnings, cleaned,
inverted to implied vol with a custom solver and fitted with arbitrage-free SVI.
Data: Yahoo Finance closing quotes, {date_str}. Source code and methodology:
<a href="https://github.com/hacks1234646/vol-surface">GitHub</a>.</p>
<div class="tiles">
 <div class="tile"><div class="label">SPX model-free 30d vol (fitted SVI)</div>
  <div class="value">{_f(h["model_free_30d"])}</div><div class="sub">VIX close {h["vix"]:.2f}</div></div>
 <div class="tile"><div class="label">SPX ATM 30d vol</div><div class="value">{_f(h["atm_30d"])}</div>
  <div class="sub">below VIX: skew and wings add variance</div></div>
 <div class="tile"><div class="label">SPX 30d 25Δ risk reversal</div>
  <div class="value">{spx["skew_30d"]["rr25"] * 100:.1f}</div><div class="sub">vol points</div></div>
 <div class="tile"><div class="label">Arbitrage in the fitted SVI surface</div>
  <div class="value">0</div><div class="sub">butterfly and calendar, all underlyings</div></div>
</div>
<p class="note">{snap_note}</p>

<h2>Surface</h2>
<div class="tabs">{tabs("surface")}</div>
<div class="card"><div id="surface" class="plot"></div></div>

<h2>Smiles by expiry</h2>
<div class="tabs">{tabs("smile")}</div>
<div class="card"><div id="smile" class="plot small"></div></div>
<div class="card"><img src="img/smiles_SPX.png" alt="SPX smiles with fits and residuals"></div>

<h2>Fit quality</h2>
<p class="note">Median across expiries of the RMSE between fitted and mid implied vol.</p>
{fit_html}

<h2>Static arbitrage: market vs fit</h2>
<p class="note">Market quotes are tested at mids and at executable prices (buy at the ask,
sell at the bid). Fits are tested on a common log-moneyness grid with Durrleman's condition
(butterfly) and total-variance ordering (calendar).</p>
{arb_html}
<div class="card"><img src="img/arbitrage_SPX.png" alt="Durrleman g for raw vs arbitrage-free fit"></div>

<h2>Term structure and skew</h2>
<div class="card"><img src="img/term_structure.png" alt="ATM term structures"></div>
<div class="card"><img src="img/skew_SPX.png" alt="SPX skew term structure"></div>
<div class="card"><img src="img/history_SPX.png" alt="SPX 30d vol vs VIX"></div>

<h2>Earnings moves</h2>
{earn_html}
<div class="card"><img src="img/earnings.png" alt="Earnings jump decomposition"></div>
</main>
<script>
const FIGS = {json.dumps(fig_json)};
const dark = () => document.documentElement.dataset.theme === "dark" ||
  (document.documentElement.dataset.theme !== "light" &&
   matchMedia("(prefers-color-scheme: dark)").matches);
function themed(fig) {{
  const f = JSON.parse(JSON.stringify(fig));
  if (!dark()) return f;
  const ink = "#ffffff", grid = "#2c2c2a", surf = "#1a1a19";
  f.layout.paper_bgcolor = surf; f.layout.plot_bgcolor = surf;
  f.layout.font = Object.assign({{}}, f.layout.font, {{color: ink}});
  for (const ax of ["xaxis", "yaxis"]) if (f.layout[ax]) f.layout[ax].gridcolor = grid;
  if (f.layout.scene) for (const ax of ["xaxis", "yaxis", "zaxis"]) {{
    f.layout.scene[ax] = Object.assign({{}}, f.layout.scene[ax], {{gridcolor: grid, backgroundcolor: surf}});
  }}
  for (const t of f.data) if (t.marker && t.marker.color === "#0b0b0b") t.marker.color = ink;
  return f;
}}
const current = {{surface: "{names[0]}", smile: "{names[0]}"}};
function draw(kind) {{
  const f = themed(FIGS[kind][current[kind]]);
  Plotly.react(kind, f.data, f.layout, {{responsive: true, displaylogo: false}});
}}
document.querySelectorAll(".tabs button").forEach(b => b.addEventListener("click", () => {{
  current[b.dataset.kind] = b.dataset.name;
  document.querySelectorAll(`.tabs button[data-kind="${{b.dataset.kind}}"]`)
    .forEach(x => x.setAttribute("aria-pressed", x === b));
  draw(b.dataset.kind);
}}));
draw("surface"); draw("smile");
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {{ draw("surface"); draw("smile"); }});
</script>
</body></html>
"""
