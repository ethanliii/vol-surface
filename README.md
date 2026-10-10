# vol-surface

Turns free, noisy end-of-day option quotes into an implied volatility surface you could actually quote and risk-manage off, then uses it to price the earnings move in four stocks reporting in October 2026.

Live site with the 3D surface and a smile explorer for every expiry: **https://hacks1234646.github.io/vol-surface/**

![SPX implied volatility surface](docs/img/surface_SPX.png)

## The problem

An options desk needs two things from its vol surface every morning.

1. **A surface it can mark the book on.** Every quote and every Greek is computed off the surface. If the surface has arbitrage in it, such as a negative probability density in the wings or two expiries crossing, then prices quoted off it can be picked off and the risk numbers are wrong.
2. **A view on event vol.** Before a company reports earnings, part of the option premium pays for the earnings-day jump. To decide whether to buy or sell that event, you first need to separate the jump from ordinary day-to-day vol.

Raw quotes are not good enough for either job. On the 2026-10-01 close, about 1,900 SPX quotes were stale, crossed, one-sided or inconsistent with their neighbours. At mid prices, 14% of SPX strike-pair and strike-triple checks show a butterfly or vertical-spread arbitrage. Fitting a smile straight through that data reproduces the problem in the model.

This project builds the whole chain from scratch: cleaning, forwards from put-call parity, an implied vol solver, an arbitrage-free SVI surface, and the analytics on top. It covers **SPX index options** (European, cash-settled) and **JPM, NFLX, TSLA and META**, using Yahoo Finance chains and US Treasury yields.

## Results (2026-10-01 close)

> The project holds one daily snapshot so far. A GitHub Actions job adds one after every close and redeploys the site, and the time-series charts fill in as history builds up.

### 1. A clean SPX surface

The arbitrage-free surface fits 8,810 SPX quotes across 48 expiries with a median error of 0.30 vol points. It has zero butterfly and zero calendar arbitrage on the test grid.

To check it against a number nobody fitted to, the project computes the model-free 30-day vol from the surface with the VIX formula. It comes out at 16.17%, against a VIX close of 16.39. A 0.22 vol-point gap means the forwards, discounting, implied vols and wing extrapolation all hold together.

What the surface shows:

- ATM vol rises from 12.1% at 7 days to 16.5% at one year.
- The 30-day 25-delta risk reversal is −3.7 vol points and the 91-day is −4.9, the usual put skew.
- ATM 30-day vol is 13.34%, about 3 points below the VIX-style number, because the VIX formula also prices the expensive downside wing.

![ATM term structures](docs/img/term_structure.png)

![SPX smiles with fits and residuals](docs/img/smiles_SPX.png)

Fit error by underlying, as the median over expiries of the RMSE between fitted and mid implied vol, in vol points:

| Underlying | Expiries | Quotes | Raw SVI | Arb-free SVI | SSVI | ATM 30d | RR25 30d |
|---|---:|---:|---:|---:|---:|---:|---:|
| SPX | 48 | 8,810 | 0.20 | 0.30 | 2.28 | 13.34% | −3.73 |
| JPM | 10 | 246 | 0.21 | 0.23 | 0.35 | 26.38% | −2.34 |
| META | 14 | 1,228 | 0.15 | 0.16 | 0.73 | 44.68% | 0.24 |
| NFLX | 11 | 385 | 0.53 | 0.54 | 1.67 | 44.90% | 0.20 |
| TSLA | 15 | 1,328 | 0.77 | 0.78 | 5.24 | 44.23% | 0.29 |

Removing arbitrage costs about 0.1 vol points of fit on SPX and at most 0.02 on the single stocks. SSVI uses only three parameters for the whole surface and cannot fit the steep short-dated SPX skew, so it is kept as a benchmark rather than used for marking.

![SPX skew term structure](docs/img/skew_SPX.png)

### 2. Pricing the earnings move

Each stock's ATM term structure is split into a base daily vol plus a one-off jump on the earnings date. The jump is the move the options market is pricing for the report.

| Stock | Earnings | Implied move (1 sd) | Expected abs. move | Two-expiry estimate | ATM straddle / F | Base vol |
|---|---|---:|---:|---:|---:|---:|
| JPM | 2026-10-13 | 2.89% | 2.31% | 3.97% | 4.86% | 24.3% |
| META | 2026-10-28 | 7.29% | 5.82% | 7.37% | 10.11% | 36.6% |
| NFLX | 2026-10-20 | 8.02% | 6.40% | 9.28% | 9.52% | 34.3% |
| TSLA | 2026-10-21 | 5.30% | 4.23% | 6.01% | 9.25% | 40.3% |

![Earnings jump decomposition](docs/img/earnings.png)

The common shortcut is to read the implied move off the straddle of the first expiry after earnings. That overstates the event, because the straddle also pays for all the ordinary vol until expiry. For META the straddle says 10.1% while the event itself is priced at 7.3%.

The two-expiry estimate uses only the expiries either side of the report, so it is noisier than the fit across all expiries within 120 days. JPM is the least reliable, because only one of its pre-earnings expiries passes the quote filters.

The trade decision is the comparison with how far each stock has actually moved on past reports. That needs a history of earnings-day returns and is the first item under next steps.

### 3. How much of the raw arbitrage is real

| Underlying | Object | Butterfly violations / tests | Calendar violations / tests |
|---|---|---:|---:|
| SPX | market quotes (mid) | 2,448 / 17,654 | 62 / 8,542 |
| SPX | market quotes (executable bid/ask) | 388 / 17,654 | 57 / 8,542 |
| SPX | raw SVI (per expiry) | 2,180 / 14,448 | 3,843 / 14,147 |
| SPX | **arbitrage-free SVI** | **0** / 14,448 | **0** / 14,147 |
| TSLA | market quotes (executable bid/ask) | 4 / 2,547 | 0 / 1,068 |
| TSLA | raw SVI (per expiry) | 348 / 4,515 | 1,100 / 4,214 |
| TSLA | **arbitrage-free SVI** | **0** / 4,515 | **0** / 4,214 |

Most of the violations at mid are adjacent 5-point SPX strikes whose mids differ by about one tick. That is quote noise, and it disappears once you have to cross the spread.

The violations that survive at bid and ask come almost entirely from stale lines in Yahoo's SPXW quarterly and month-end series. Removing the stalest quote in each one (134 SPX quotes) leaves no executable arbitrage in any underlying.

Unconstrained SVI fits sit close to the quotes, but they put arbitrage in the extrapolated wings and between neighbouring expiries. The constrained fit removes it.

![Durrleman's condition for raw vs arbitrage-free SVI](docs/img/arbitrage_SPX.png)

The full tables for all five underlyings are on the [site](https://hacks1234646.github.io/vol-surface/) and in [`docs/results.json`](docs/results.json).

## How it works

1. **Collect.** After each close, save the raw option chains, the Treasury curve and the VIX close as parquet (about 0.8 MB a day).
2. **Clean.** Drop zero bids, crossed and very wide markets, stale lines, parity and IV outliers, and quotes that form an executable arbitrage. Every dropped quote keeps its reason, so the cleaning can be audited.
3. **Forward and discount factor.** A weighted regression of call minus put on strike gives both, so no dividend model is needed. The SPX parity-implied rate runs about 58 bp above Treasuries beyond 60 days, in line with where SPX box spreads finance.
4. **Implied vol.** A vectorized Newton solver on log price, with a bisection fallback so it always converges. It inverts 200,000 options in under a second.
5. **Fit.** Raw SVI per expiry, then an arbitrage-free SVI surface with the butterfly and calendar conditions imposed as constraints. SSVI is fitted as a benchmark. The 48-expiry SPX surface takes about 4 seconds.
6. **Analytics.** ATM term structure, skew, model-free 30-day vol and the earnings decomposition.

[METHODOLOGY.md](METHODOLOGY.md) has the formulas, the cleaning table with counts, the calibration details and what the 47 unit tests cover.

## Run it

```bash
git clone https://github.com/hacks1234646/vol-surface && cd vol-surface
make install   # venv, dependencies and the package (Python 3.11+)
make build     # fit and analyse every snapshot in data/, write figures and the site to docs/
make collect   # fetch today's snapshot (run after the close)
make test      # 47 unit tests
```

The CLI is `volsurface {collect,process,build}`.

## Limitations

- **Data.** Yahoo quotes are free, delayed and have no timestamps, so stale quotes can only be caught by inconsistency with their neighbours. A professional feed such as Cboe DataShop or OptionMetrics would make most of the cleaning unnecessary. Yahoo also serves no historical chains, so a missed day cannot be backfilled.
- **History.** One snapshot so far. Anything about how skew or the VIX gap behaves over time needs the history the daily job is collecting.
- **American exercise.** Single-stock options are American but are inverted with Black-76. Using only out-of-the-money options keeps the early-exercise premium small, not zero.
- **SVI's shape.** SVI cannot bend sharply enough at the money for TSLA, so the fit sits about 1 vol point below the market there. The earnings decomposition uses market ATM variance for that reason.
- **Inputs.** Treasury par yields stand in for zero rates, and Yahoo's earnings dates can be estimates with no before-open or after-close timing.

## Next steps

- Compare each implied earnings move with the stock's realized moves on past reports, which is the actual buy-or-sell signal.
- With more history, measure the variance risk premium (implied vs later realized variance) and whether skew moves sticky-strike or sticky-delta.
- De-Americanize single-stock prices with a binomial tree before inverting.
- A local-vol surface (Dupire) from the arbitrage-free fit.

## Project layout

```
src/volsurface/
  collect.py        yfinance snapshots, Treasury/FRED rates, VIX
  rates.py          rate curve
  black_scholes.py  Black-Scholes / Black-76 prices and Greeks
  iv.py             vectorized implied-vol solver
  cleaning.py       quote filters, parity forward and discount factor, OTM IVs
  svi.py            raw SVI, Durrleman's condition, quasi-explicit and arbitrage-free calibration
  ssvi.py           SSVI surface
  surface.py        assembling slices into a surface, interpolation
  arbitrage.py      butterfly / calendar diagnostics for quotes and fits
  analysis.py       term structure, skew, model-free vol, earnings moves
  pipeline.py       per-snapshot processing and caching (data/derived/)
  plots.py, report.py  figures and the GitHub Pages site (docs/)
tests/              unit and end-to-end tests
.github/workflows/  CI, daily collection, Pages deployment
```

## References

- J. Gatheral, *The Volatility Surface* (2006).
- J. Gatheral and A. Jacquier, "Arbitrage-free SVI volatility surfaces", *Quantitative Finance* (2014).
- Zeliade Systems, "Quasi-explicit calibration of Gatheral's SVI model" (2009).
- R. Lee, "The moment formula for implied volatility at extreme strikes", *Mathematical Finance* (2004).
- Cboe, *VIX White Paper*.
