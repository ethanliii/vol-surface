# Methodology

How each step of the pipeline in the [README](README.md) works, in the order the code runs.

## 1. Data and time

- **SPX vs SPY:** Yahoo serves SPX index options (`^SPX`, roots SPX and SPXW), so the index work uses European options and avoids SPY's American-exercise bias. SPY is not needed.
- **Snapshots** are stored as raw quotes (parquet, about 0.8 MB/day) under `data/snapshots/<date>/`, together with the Treasury curve and the VIX close. All cleaning is downstream, so it can be changed and the full history re-processed.
- **Time to expiry** is ACT/365 in calendar time, from the 16:00 ET close to settlement. Settlement is 09:30 ET for AM-settled SPX monthlies and 16:00 ET otherwise. When an expiry has both SPX (AM) and SPXW (PM) series, the one with more two-sided quotes is kept.
- **Rates:** Treasury par yields (FRED as fallback), converted from bond-equivalent to continuous compounding, r = 2 ln(1 + y/2), and interpolated linearly. They are only a prior: discount factors come from parity wherever they can (next section).

## 2. Cleaning

Every quote keeps a `drop_reason`, so the cleaning is auditable (SPX counts on 2026-10-01 in brackets):

| Filter | Rule | SPX |
|---|---|---:|
| zero bid | bid ≤ 0 | 749 |
| crossed / locked | ask ≤ bid | 13 |
| wide market | (ask − bid) / mid > 50% | 492 |
| stale line | no trade in 90 days (Yahoo has no quote timestamps; catches e.g. NFLX pre-split strikes still showing 2025 markets) | 441 |
| in the money | only OTM options relative to the parity forward are used (puts below F, calls above) | 5,349 |
| parity outlier | strike pair inconsistent with the parity regression | 36 |
| IV outlier | more than max(3 × IV half-spread, 2 vol pts) from the rolling median of neighbouring strikes | 28 |
| executable arbitrage | quote forms a tradeable butterfly / vertical-spread arbitrage with its neighbours; the stalest quote is removed until none remain | 134 |
| **used** | | **8,810** |

## 3. Forward and discount factor from put-call parity

For each expiry, C − P = D·F − D·K. A weighted least-squares fit of (C − P) on K over near-the-money strikes gives the discount factor D (minus the slope) and the forward F (intercept / D), with weights equal to the inverse squared bid-ask widths:

1. **Robust pre-screen.** Per-strike forwards K + (C − P)/D are compared with their median, using a MAD-scaled tolerance. A handful of stale lines otherwise drags the regression.
2. **When D is estimated.** For European options with at least 30 days to expiry, D is taken from the regression if the implied rate is within 200 bp of the curve. Otherwise D comes from the Treasury curve and only F is estimated.
3. **American options (single stocks).** Parity is only an inequality for American options, so D always comes from the curve. F comes from the strikes nearest the money, where the early-exercise premium is smallest.

Findings:

- **SPX financing above Treasuries.** The SPX parity-implied rate runs about 58 bp above Treasury CMT yields beyond 60 days, consistent with the known premium of SPX box-spread financing over Treasuries.
- **Discrete dividends.** JPM's short-dated forwards show its October ex-dividend date (negative implied carry), with no dividend model needed.
- **Parity fit quality.** Median parity residual for SPX: 0.22 index points.

## 4. Black-Scholes and implied volatility

Forward (Black-76) form, with log-moneyness k = ln(K/F) and total variance w = σ²T:

$$C = D\,[F\,N(d_1) - K\,N(d_2)],\qquad d_{1,2} = \frac{-k}{\sqrt{w}} \pm \frac{\sqrt{w}}{2}$$

The spot form with continuous dividend yield and all Greeks (delta, gamma, vega, theta, rho, vanna, volga) are implemented in `black_scholes.py`.

**Implied vol solver** (`iv.py`), vectorized over a whole chain:

1. Convert in-the-money prices to the equivalent OTM price by parity, which removes the intrinsic value that would swamp the time value.
2. Check the no-arbitrage bounds 0 < p < min(1, e^k) in units of the forward; anything outside returns NaN.
3. Run Newton on **log price** in total vol s = σ√T. In the wings the price is tiny and convex, and log space keeps Newton fast there.
4. Keep a bracket around the root (the price is increasing in s) and fall back to bisection whenever a Newton step would leave it, so convergence is guaranteed.

It inverts 200,000 random options in under a second, to within the floating-point conditioning bound.

## 5. Smile and surface fitting

**Raw SVI** per expiry:

$$w(k) = a + b\left[\rho\,(k-m) + \sqrt{(k-m)^2+\sigma^2}\right]$$

The parameter constraints are:

- b ≥ 0, |ρ| < 1, σ > 0;
- non-negative minimum variance, a + bσ√(1−ρ²) ≥ 0;
- **Roger Lee's moment bound** on the wing slopes, b(1 ± ρ) ≤ 2.

**Stage 1: quasi-explicit calibration** (Zeliade 2009). For fixed (m, σ), SVI is *linear* in (a, u, v), where u = bσ(1+ρ) and v = bσ(1−ρ). Lee's bound becomes the box 0 ≤ u, v ≤ 2σ, so the inner problem is a bounded linear least-squares solve and only (m, σ) is searched (grid, then Nelder-Mead).

Residuals are weighted by each quote's bid-ask uncertainty expressed in total variance (dw = 2σT·dσ), so the objective is approximately Σ((σ_model − σ_mid)/half-spread)².

**Stage 2: arbitrage-free SVI.**

- **Butterfly:** Durrleman's condition must hold on a dense grid,

$$g(k) = \Big(1 - \frac{k\,w'}{2w}\Big)^2 - \frac{w'^2}{4}\Big(\frac{1}{w}+\frac{1}{4}\Big) + \frac{w''}{2} \;\ge\; 0,$$

  where g(k) is, up to a positive factor, the risk-neutral density.
- **Calendar:** the slice must stay between its fitted neighbours, w_{T₁}(k) ≤ w_T(k) ≤ w_{T₂}(k), with ordered wing slopes so slices cannot cross in the extrapolated wings.
- **How each constraint is imposed:** the calendar constraints are linear in (a, u, v) and enter the inner problem as a small convex QP (Goldfarb-Idnani via `quadprog`). Durrleman's condition enters the outer search as an exact L1 penalty, followed by a final SLSQP polish if needed.
- **Fitting order:** slices are fitted in order of strike coverage, widest first. Data-rich expiries pin down the wings, and narrowly quoted expiries are sandwiched between fitted neighbours instead of letting their extrapolated wings constrain the rest.

The full 48-expiry SPX surface fits in about 4 seconds. Between expiries, total variance is interpolated linearly in T at fixed k, which preserves the calendar ordering.

**SSVI** (Gatheral & Jacquier 2014) as a global benchmark:

$$w(k,\theta_t) = \frac{\theta_t}{2}\Big(1 + \rho\varphi k + \sqrt{(\varphi k + \rho)^2 + 1 - \rho^2}\Big),\qquad \varphi(\theta) = \frac{\eta}{\theta^\gamma(1+\theta)^{1-\gamma}}$$

- θ_t is the market ATM total variance, made monotone by isotonic regression.
- The constraints 0 < γ ≤ ½ and η(1 + |ρ|) ≤ 2 are sufficient for no static arbitrage.

## 6. Arbitrage checks (`arbitrage.py`)

- **Quotes, one expiry.** OTM quotes are converted to call prices by parity and checked on consecutive strikes. Calls must be decreasing in strike, with slope ≥ −1 per unit forward, and convex: the butterfly λC(K₁) + (1−λ)C(K₃) − C(K₂) must be ≥ 0. Each check is run at mids and at executable prices (wings bought at the ask, body sold at the bid).
- **Quotes, across expiries.** At the longer expiry's strikes inside the shorter expiry's quoted range, w(k, T₂) ≥ w(k, T₁) must hold. It is checked at mids, and executably by comparing the longer expiry's ask with the shorter expiry's bid.
- **Fits.** Durrleman's g ≥ 0 and total variance ordered in T, both on a common grid spanning every expiry's strikes plus wings. Violations below 10⁻⁶ in total variance are treated as numerical noise; that is about 0.0003 vol points at one year.

## 7. Analytics (`analysis.py`)

- **ATM vol** at standard tenors comes from the arbitrage-free surface (at-the-forward, k = 0). **Skew** is measured three ways: the 25-delta risk reversal and butterfly (forward deltas found by root-finding on the fitted smile), σ(0.9F) − σ(1.1F), and dσ/dk at the money.
- **Model-free variance** of one smile, the continuous-strike VIX formula, with q(k) the undiscounted OTM price per unit forward:

$$\sigma^2_{\text{MF}}\,T = 2\int q(k)\,e^{-k}\,dk$$

  It is interpolated linearly in total variance between the two expiries bracketing 30 days.
- **Earnings move.** The market ATM total variance of each expiry is decomposed as

$$w(T) = \sigma_b^2\,\tau(T) + J^2\,\mathbf{1}\{T > t_E\}$$

  where τ is trading-day time on the NYSE calendar, so the base variance is not smeared over weekends. The model is fitted by non-negative least squares over expiries within 120 days. J is the standard deviation of the earnings-day log return, and E|move| = J√(2/π).

## Tests

47 unit tests (`pytest`) cover every piece of pricing and fitting math:

- **Black-Scholes:** Black-Scholes and Black-76 prices against Hull and Haug textbook values; Greeks against Hull and against finite differences; put-call parity on random inputs.
- **Implied vol:** round-trips across strikes, maturities and vol levels, plus 200,000 random options checked against the conditioning bound, and NaN for arbitrage-violating prices.
- **SVI:** derivatives; Durrleman's g equal to the density from finite-differenced call prices; Vogt's arbitrageable SVI example detected and repaired; calendar constraints respected; surface interpolation monotone in T.
- **SSVI:** random parameters satisfying the constraints produce no arbitrage; parameter recovery.
- **Cleaning:** synthetic chains with known forward and discount factor; each filter triggered.
- **Arbitrage:** each violation type detected; stale-quote repair.
- **Analytics:** model-free variance equals σ²T for a flat smile; closed-form delta strikes; jump recovery.
- **End to end:** quotes generated from a known arbitrage-free SSVI surface go through the whole pipeline, and the known ATM vols and ρ are recovered.
