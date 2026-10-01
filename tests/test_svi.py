import numpy as np
import pytest

from volsurface.black_scholes import normalized_price
from volsurface.surface import Slice, SVISurface
from volsurface.svi import (
    BUTTERFLY_TOL,
    CALENDAR_TOL,
    SVIParams,
    fit_svi_arbitrage_free,
    fit_svi_raw,
    svi_derivatives,
    svi_g,
    svi_total_variance,
)

SPX_LIKE = SVIParams(a=0.002, b=0.08, rho=-0.75, m=0.05, sigma=0.12)
# Axel Vogt's example from Gatheral & Jacquier (2014): valid raw SVI parameters
# whose smile nevertheless admits butterfly arbitrage.
VOGT = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)


def test_derivatives_match_finite_differences() -> None:
    k = np.linspace(-1, 1, 41)
    w, w1, w2 = svi_derivatives(SPX_LIKE, k)
    h = 1e-5
    f = lambda x: svi_total_variance(SPX_LIKE, x)  # noqa: E731
    np.testing.assert_allclose(w, f(k))
    np.testing.assert_allclose(w1, (f(k + h) - f(k - h)) / (2 * h), rtol=1e-6)
    np.testing.assert_allclose(w2, (f(k + h) - 2 * f(k) + f(k - h)) / h**2, rtol=1e-3)


def test_durrleman_g_is_the_risk_neutral_density() -> None:
    """density of x = ln(S_T/F): p(k) = g(k) / sqrt(2 pi w) exp(-d2^2 / 2)."""
    k = np.linspace(-0.6, 0.4, 21)
    w = svi_total_variance(SPX_LIKE, k)
    d2 = -k / np.sqrt(w) - np.sqrt(w) / 2
    p_formula = svi_g(SPX_LIKE, k) / np.sqrt(2 * np.pi * w) * np.exp(-0.5 * d2**2)

    def call(K):  # undiscounted call per unit forward, F = 1
        kk = np.log(K)
        return normalized_price(kk, np.sqrt(svi_total_variance(SPX_LIKE, kk)), True)

    K, h = np.exp(k), 1e-4
    d2c = (call(K + h) - 2 * call(K) + call(K - h)) / h**2
    np.testing.assert_allclose(p_formula, K * d2c, rtol=1e-4, atol=1e-6)


def test_vogt_example_has_butterfly_arbitrage() -> None:
    assert VOGT.is_valid()
    k = np.linspace(-1.5, 1.5, 601)
    assert svi_g(VOGT, k).min() < -0.03  # g dips to about -0.033 near k = 0.7


def test_raw_fit_recovers_smile() -> None:
    k = np.linspace(-0.5, 0.3, 40)
    w = svi_total_variance(SPX_LIKE, k)
    p = fit_svi_raw(k, w)
    np.testing.assert_allclose(svi_total_variance(p, k), w, atol=1e-7)
    assert p.is_valid()


def test_raw_fit_respects_parameter_constraints_under_noise() -> None:
    rng = np.random.default_rng(3)
    k = np.linspace(-0.4, 0.2, 30)
    w = svi_total_variance(SPX_LIKE, k) * (1 + 0.02 * rng.standard_normal(k.size))
    p = fit_svi_raw(k, w)
    assert p.is_valid()
    assert p.b * (1 + abs(p.rho)) <= 2 + 1e-9


def test_arbitrage_free_fit_removes_butterfly_arbitrage() -> None:
    k = np.linspace(-1.0, 1.0, 50)
    w = svi_total_variance(VOGT, k)
    grid = np.linspace(-2, 2, 401)
    p, info = fit_svi_arbitrage_free(k, w, k_grid=grid)
    assert info.feasible
    assert svi_g(p, grid).min() >= -BUTTERFLY_TOL
    # Still close to the data: the repair is a small perturbation.
    iv_err = np.sqrt(svi_total_variance(p, k)) - np.sqrt(w)
    assert np.sqrt(np.mean(iv_err**2)) < 0.01


def test_arbitrage_free_fit_respects_calendar_neighbours() -> None:
    lower = SVIParams(0.01, 0.10, -0.6, 0.0, 0.15)
    upper = SVIParams(0.03, 0.14, -0.6, 0.0, 0.20)
    k = np.linspace(-0.6, 0.4, 40)
    # Target data that dips below the shorter slice in the left wing.
    target = svi_total_variance(SVIParams(0.018, 0.08, -0.4, 0.05, 0.1), k)
    assert np.any(target < svi_total_variance(lower, k))
    grid = np.linspace(-2, 2, 301)
    p, info = fit_svi_arbitrage_free(k, target, lower=lower, upper=upper, k_grid=grid)
    w = svi_total_variance(p, grid)
    assert info.feasible
    assert np.all(w >= svi_total_variance(lower, grid) - CALENDAR_TOL)
    assert np.all(w <= svi_total_variance(upper, grid) + CALENDAR_TOL)
    assert svi_g(p, grid).min() >= -BUTTERFLY_TOL


def test_surface_interpolation_is_calendar_monotone() -> None:
    p1 = SVIParams(0.002, 0.05, -0.7, 0.0, 0.1)
    p2 = SVIParams(0.010, 0.09, -0.7, 0.0, 0.15)
    mk = lambda T, p: Slice(  # noqa: E731
        expiry=None,
        T=T,
        F=1.0,
        k=np.array([0.0]),
        w=np.array([1.0]),
        iv=np.array([1.0]),
        iv_bid=np.array([1.0]),
        iv_ask=np.array([1.0]),
        weights=np.array([1.0]),
        free=p,
    )
    surf = SVISurface([mk(0.1, p1), mk(0.5, p2)], np.linspace(-1, 1, 11))
    k = np.linspace(-1, 1, 11)
    Ts = np.linspace(0.01, 1.0, 60)
    W = np.array([surf.total_variance(k, T) for T in Ts])
    assert np.all(np.diff(W, axis=0) >= -1e-14)
    np.testing.assert_allclose(surf.total_variance(k, 0.5), svi_total_variance(p2, k))
    assert surf.atm_vol(0.1) == pytest.approx(np.sqrt(svi_total_variance(p1, 0.0) / 0.1))
