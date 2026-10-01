import numpy as np
import pytest

from volsurface.black_scholes import black76_price, black76_vega, bs_price, normalized_price
from volsurface.iv import implied_total_vol, implied_vol, implied_vol_bs


def test_known_value_round_trip() -> None:
    price = bs_price(42, 40, 0.5, 0.10, 0.20, is_call=True)
    assert implied_vol_bs(price, 42, 40, 0.5, 0.10) == pytest.approx(0.20, abs=1e-12)


@pytest.mark.parametrize("is_call", [True, False])
def test_round_trip_grid(is_call: bool) -> None:
    """IV(price(sigma)) == sigma across strikes, maturities and vol levels."""
    T = np.array([1 / 365, 7 / 365, 30 / 365, 0.25, 1.0, 3.0, 10.0])
    k = np.linspace(-1.5, 1.0, 41)
    vol = np.array([0.03, 0.1, 0.2, 0.5, 1.0, 2.5])
    TT, KK, VV = np.meshgrid(T, k, vol, indexing="ij")
    s = VV * np.sqrt(TT)
    F, D = 100.0, 0.97
    K = F * np.exp(KK)
    price = black76_price(F, K, TT, VV, D, is_call)
    # Only test prices that carry vol information in double precision: deep OTM
    # prices below ~1e-11 of the forward are indistinguishable from zero.
    intrinsic = D * np.maximum((F - K) if is_call else (K - F), 0)
    informative = (price - intrinsic > 1e-11 * F) & (s < 6)
    iv = implied_vol(price, F, K, TT, D, is_call)
    assert np.isfinite(iv[informative]).all()
    # Achievable accuracy is limited by the conditioning of the inversion:
    # a price rounding error of ~1e-15 * price moves the vol by that / vega.
    vega = black76_vega(F, K, TT, VV, D)
    with np.errstate(divide="ignore", invalid="ignore"):
        tol = 1e-9 * VV + 1e-13 * price / vega
    err = np.abs(iv - VV)
    assert np.all(err[informative] <= tol[informative])
    # And in price space the round trip is at machine-level precision.
    reprice = black76_price(F, K, TT, iv, D, is_call)
    np.testing.assert_allclose(reprice[informative], price[informative], rtol=1e-9, atol=1e-13)


def test_calls_and_puts_give_same_vol() -> None:
    F, D, T = 5000.0, 0.95, 0.8
    K = np.linspace(3000, 7000, 81)
    vol = 0.15 + 0.1 * (np.log(K / F)) ** 2
    c = black76_price(F, K, T, vol, D, True)
    p = black76_price(F, K, T, vol, D, False)
    np.testing.assert_allclose(implied_vol(c, F, K, T, D, True), vol, rtol=1e-8)
    np.testing.assert_allclose(implied_vol(p, F, K, T, D, False), vol, rtol=1e-8)


def test_arbitrage_violating_prices_return_nan() -> None:
    k = np.array([0.0, 0.1, -0.1, 0.1, 0.0])
    p = np.array([-0.01, 1.2, 0.0, 0.0, 1.0])  # negative, above bound, zero, zero, at bound
    assert np.isnan(implied_total_vol(p, k, True)).all()
    # ITM call below intrinsic -> no IV
    assert np.isnan(implied_total_vol(0.05, -0.1, True))  # intrinsic is 1 - e^-0.1 = 0.095


def test_extreme_wings_converge() -> None:
    k = np.array([-3.0, -2.0, 2.0, 3.0])
    s = np.array([0.9, 0.7, 0.8, 1.2])
    is_call = k > 0
    p = normalized_price(k, s, is_call)
    np.testing.assert_allclose(implied_total_vol(p, k, is_call), s, rtol=1e-10)


def test_vectorized_shapes() -> None:
    p = normalized_price(np.zeros((3, 4)), 0.2 * np.ones((3, 4)))
    out = implied_total_vol(p, 0.0)
    assert out.shape == (3, 4)
    np.testing.assert_allclose(out, 0.2, rtol=1e-12)


def test_random_options_accuracy_within_conditioning() -> None:
    """200k random (k, s, call/put): error is within the floating-point
    conditioning bound wherever an IV is returned."""
    rng = np.random.default_rng(1)
    n = 200_000
    k = rng.uniform(-1, 1, n)
    s = rng.uniform(0.01, 1.5, n)
    is_call = rng.random(n) < 0.5
    p = normalized_price(k, s, is_call)
    out = implied_total_vol(p, k, is_call)
    ok = np.isfinite(out)
    assert ok.mean() > 0.97
    from volsurface.black_scholes import normalized_vega

    bound = 1e-9 * s + 1e-13 * p / np.maximum(normalized_vega(k, s), 1e-300)
    assert np.all(np.abs(out - s)[ok] <= bound[ok])
