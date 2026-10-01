import numpy as np
import pytest

from volsurface.black_scholes import (
    black76_delta,
    black76_price,
    black76_vega,
    bs_greeks,
    bs_price,
    normalized_price,
    normalized_vega,
)


def test_hull_textbook_values() -> None:
    # Hull, Options Futures and Other Derivatives, Example 15.6: c = 4.76, p = 0.81.
    c = bs_price(42, 40, 0.5, 0.10, 0.20, is_call=True)
    p = bs_price(42, 40, 0.5, 0.10, 0.20, is_call=False)
    assert c == pytest.approx(4.7594, abs=1e-4)
    assert p == pytest.approx(0.8086, abs=1e-4)


def test_haug_generalized_bsm_put() -> None:
    # Haug, Complete Guide to Option Pricing Formulas: S=75, K=70, T=0.5,
    # r=10%, b=r-q=5%, sigma=35% -> put = 4.0870.
    p = bs_price(75, 70, 0.5, 0.10, 0.35, q=0.05, is_call=False)
    assert p == pytest.approx(4.0870, abs=1e-4)


def test_haug_black76() -> None:
    # Haug: F=19, K=19, T=0.75, r=10%, sigma=28% -> call = put = 1.7011.
    D = np.exp(-0.10 * 0.75)
    c = black76_price(19, 19, 0.75, 0.28, D, True)
    p = black76_price(19, 19, 0.75, 0.28, D, False)
    assert c == pytest.approx(1.7011, abs=1e-4)
    assert p == pytest.approx(c, abs=1e-12)


def test_hull_greeks() -> None:
    # Hull Ch. 19 running example: S=49, K=50, r=5%, sigma=20%, T=20 weeks.
    g = bs_greeks(49, 50, 0.3846, 0.05, 0.20, is_call=True)
    assert g["delta"] == pytest.approx(0.522, abs=1e-3)
    assert g["gamma"] == pytest.approx(0.066, abs=1e-3)
    assert g["vega"] == pytest.approx(12.1, abs=0.05)
    assert g["theta"] == pytest.approx(-4.31, abs=0.01)
    assert g["rho"] == pytest.approx(8.91, abs=0.01)


@pytest.mark.parametrize("is_call", [True, False])
def test_greeks_match_finite_differences(is_call: bool) -> None:
    S, K, T, r, q, vol = 100.0, 95.0, 0.7, 0.03, 0.015, 0.27
    g = bs_greeks(S, K, T, r, vol, q, is_call)

    def f(**kw):
        args = dict(S=S, K=K, T=T, r=r, sigma=vol, q=q, is_call=is_call)
        args.update(kw)
        return bs_price(**args)

    h = 1e-4
    assert g["delta"] == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-6)
    assert g["gamma"] == pytest.approx((f(S=S + 1e-2) - 2 * f() + f(S=S - 1e-2)) / 1e-4, rel=1e-4)
    assert g["vega"] == pytest.approx((f(sigma=vol + h) - f(sigma=vol - h)) / (2 * h), rel=1e-6)
    assert g["theta"] == pytest.approx(-(f(T=T + h) - f(T=T - h)) / (2 * h), rel=1e-6)
    assert g["rho"] == pytest.approx((f(r=r + h) - f(r=r - h)) / (2 * h), rel=1e-6)
    dvega = bs_greeks(S, K, T, r, vol + h, q)["vega"] - bs_greeks(S, K, T, r, vol - h, q)["vega"]
    assert g["volga"] == pytest.approx(dvega / (2 * h), rel=1e-5)
    ddelta = (
        bs_greeks(S, K, T, r, vol + h, q, is_call)["delta"]
        - bs_greeks(S, K, T, r, vol - h, q, is_call)["delta"]
    )
    assert g["vanna"] == pytest.approx(ddelta / (2 * h), rel=1e-5)


def test_put_call_parity_spot_form() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    S = rng.uniform(50, 150, n)
    K = rng.uniform(40, 200, n)
    T = rng.uniform(0.01, 3, n)
    r = rng.uniform(-0.01, 0.08, n)
    q = rng.uniform(0, 0.05, n)
    vol = rng.uniform(0.05, 1.5, n)
    c = bs_price(S, K, T, r, vol, q, True)
    p = bs_price(S, K, T, r, vol, q, False)
    np.testing.assert_allclose(c - p, S * np.exp(-q * T) - K * np.exp(-r * T), atol=1e-10)


def test_black76_parity_and_bsm_consistency() -> None:
    S, K, T, r, q, vol = 100.0, 110.0, 1.3, 0.04, 0.01, 0.22
    F, D = S * np.exp((r - q) * T), np.exp(-r * T)
    for is_call in (True, False):
        assert black76_price(F, K, T, vol, D, is_call) == pytest.approx(
            bs_price(S, K, T, r, vol, q, is_call), rel=1e-12
        )
    c, p = black76_price(F, K, T, vol, D, True), black76_price(F, K, T, vol, D, False)
    assert c - p == pytest.approx(D * (F - K), abs=1e-12)


def test_black76_greeks_vs_finite_differences() -> None:
    F, K, T, vol, D = 100.0, 90.0, 0.5, 0.3, 0.98
    h = 1e-5
    vega_fd = (black76_price(F, K, T, vol + h, D) - black76_price(F, K, T, vol - h, D)) / (2 * h)
    delta_fd = (black76_price(F + h, K, T, vol, D) - black76_price(F - h, K, T, vol, D)) / (2 * h)
    assert black76_vega(F, K, T, vol, D) == pytest.approx(vega_fd, rel=1e-7)
    assert black76_delta(F, K, T, vol, D) == pytest.approx(delta_fd, rel=1e-7)


def test_normalized_price_limits_and_vega() -> None:
    k = np.array([-0.5, 0.0, 0.5])
    # s -> 0: intrinsic value; s -> inf: call -> 1, put -> e^k.
    np.testing.assert_allclose(normalized_price(k, 0.0, True), np.maximum(1 - np.exp(k), 0))
    np.testing.assert_allclose(normalized_price(k, 40.0, True), 1.0, atol=1e-12)
    np.testing.assert_allclose(normalized_price(k, 40.0, False), np.exp(k), atol=1e-12)
    # Prices increase in s and vega matches the finite difference.
    s = np.linspace(0.05, 2, 50)
    for kk in k:
        pr = normalized_price(kk, s, True)
        assert np.all(np.diff(pr) > 0)
        fd = (normalized_price(kk, s + 1e-6) - normalized_price(kk, s - 1e-6)) / 2e-6
        np.testing.assert_allclose(normalized_vega(kk, s), fd, rtol=1e-6, atol=1e-9)
