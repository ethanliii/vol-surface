import numpy as np
import pytest

from volsurface.ssvi import (
    SSVIParams,
    fit_ssvi,
    isotonic_increasing,
    ssvi_derivatives,
    ssvi_g,
    ssvi_total_variance,
)


def test_derivatives_match_finite_differences() -> None:
    p, theta = SSVIParams(-0.6, 1.2, 0.4), 0.04
    k = np.linspace(-1, 1, 21)
    w, w1, w2 = ssvi_derivatives(p, k, theta)
    f = lambda x: ssvi_total_variance(p, x, theta)  # noqa: E731
    h = 1e-5
    np.testing.assert_allclose(w, f(k))
    np.testing.assert_allclose(w1, (f(k + h) - f(k - h)) / (2 * h), rtol=1e-6)
    np.testing.assert_allclose(w2, (f(k + h) - 2 * f(k) + f(k - h)) / h**2, rtol=1e-3)


def test_atm_total_variance_equals_theta() -> None:
    p = SSVIParams(-0.7, 1.0, 0.3)
    for theta in (0.001, 0.04, 0.3):
        assert ssvi_total_variance(p, 0.0, theta) == pytest.approx(theta)


def test_parameter_constraints_imply_no_static_arbitrage() -> None:
    rng = np.random.default_rng(7)
    k = np.linspace(-3, 3, 601)
    thetas = np.geomspace(1e-4, 1.0, 25)
    for _ in range(40):
        rho = rng.uniform(-0.99, 0.99)
        eta = rng.uniform(0.01, 2 / (1 + abs(rho)))
        gamma = rng.uniform(0.01, 0.5)
        p = SSVIParams(rho, eta, gamma)
        assert p.satisfies_no_arbitrage()
        W = np.array([ssvi_total_variance(p, k, th) for th in thetas])
        assert np.all(np.diff(W, axis=0) >= -1e-14)  # calendar
        for th in thetas:
            assert ssvi_g(p, k, th).min() >= -1e-10  # butterfly


def test_isotonic_regression() -> None:
    y = np.array([1.0, 3.0, 2.0, 4.0, 3.5, 3.0, 5.0])
    out = isotonic_increasing(y)
    assert np.all(np.diff(out) >= 0)
    np.testing.assert_allclose(out, [1, 2.5, 2.5, 3.5, 3.5, 3.5, 5])
    np.testing.assert_allclose(isotonic_increasing([1.0, 2.0, 3.0]), [1, 2, 3])


def test_fit_recovers_parameters() -> None:
    true = SSVIParams(-0.65, 0.9, 0.35)
    T = np.array([0.05, 0.1, 0.25, 0.5, 1.0])
    theta = 0.03 * T + 0.0005
    ks = [np.linspace(-0.5, 0.3, 25) * np.sqrt(t) * 3 for t in T]
    ws = [ssvi_total_variance(true, k, th) for k, th in zip(ks, theta, strict=True)]
    fit = fit_ssvi(ks, ws, [np.ones_like(k) for k in ks], T, theta)
    assert fit.params.rho == pytest.approx(true.rho, abs=1e-4)
    assert fit.params.eta == pytest.approx(true.eta, rel=1e-3)
    assert fit.params.gamma == pytest.approx(true.gamma, abs=1e-3)
