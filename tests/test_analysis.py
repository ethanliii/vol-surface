from datetime import date

import numpy as np
import pytest
from scipy.stats import norm

from volsurface.analysis import (
    business_years,
    delta_strike,
    earnings_move,
    model_free_variance,
    skew_metrics,
)
from volsurface.surface import Slice, SVISurface
from volsurface.svi import SVIParams


def flat_surface(vol: float, Ts=(0.05, 0.1, 0.5, 1.0)) -> SVISurface:
    one = np.array([1.0])
    slices = [
        Slice(None, T, 1.0, one * 0, one, one, one, one, one,
              free=SVIParams(vol**2 * T, 0.0, 0.0, 0.0, 0.1))
        for T in Ts
    ]  # fmt: skip
    return SVISurface(slices, np.linspace(-1, 1, 11))


def test_model_free_variance_flat_smile_equals_sigma2_T() -> None:
    for vol, T in ((0.2, 0.1), (0.5, 1.0), (0.12, 30 / 365)):
        p = SVIParams(vol**2 * T, 0.0, 0.0, 0.0, 0.1)
        assert model_free_variance(p) == pytest.approx(vol**2 * T, rel=1e-6)


def test_model_free_variance_exceeds_atm_with_skew() -> None:
    T = 30 / 365
    p = SVIParams(a=0.0005, b=0.03, rho=-0.8, m=0.02, sigma=0.05)
    atm_w = p.a + p.b * (p.rho * -p.m + np.sqrt(p.m**2 + p.sigma**2))
    assert model_free_variance(p) > atm_w
    assert np.sqrt(model_free_variance(p) / T) < 1.0


def test_delta_strike_flat_vol_closed_form() -> None:
    vol, T = 0.25, 0.5
    surf = flat_surface(vol)
    s = vol * np.sqrt(T)
    # N(d1) = 0.25 with d1 = -k/s + s/2  =>  k = s (s/2 - N^-1(0.25))
    assert delta_strike(surf, T, 0.25) == pytest.approx(s * (s / 2 - norm.ppf(0.25)), abs=1e-9)
    assert delta_strike(surf, T, -0.25) == pytest.approx(s * (s / 2 - norm.ppf(0.75)), abs=1e-9)
    m = skew_metrics(surf, T)
    assert m["rr25"] == pytest.approx(0.0, abs=1e-12)
    assert m["atm"] == pytest.approx(vol)


def test_business_years_skips_weekends_and_holidays() -> None:
    assert business_years(date(2026, 11, 20), date(2026, 11, 30)) * 252 == pytest.approx(5)
    assert business_years(date(2026, 12, 23), date(2027, 1, 4)) * 252 == pytest.approx(6)


def test_earnings_move_recovers_jump() -> None:
    trade = date(2026, 10, 1)
    edate = date(2026, 10, 20)
    expiries = [date(2026, 10, d) for d in (9, 16, 23, 30)] + [date(2026, 11, 20)]
    base, jump = 0.30, 0.06
    tau = np.array([business_years(trade, e) for e in expiries])
    w = base**2 * tau + jump**2 * np.array([e > edate for e in expiries])
    mv = earnings_move(trade, edate, expiries, w)
    assert mv is not None
    assert mv.jump_sd == pytest.approx(jump, rel=1e-8)
    assert mv.base_vol == pytest.approx(base, rel=1e-8)
    assert mv.two_expiry_jump_sd == pytest.approx(jump, rel=1e-8)
    assert mv.expected_abs_move == pytest.approx(jump * np.sqrt(2 / np.pi))
    assert mv.straddle_move > mv.expected_abs_move  # includes diffusion
