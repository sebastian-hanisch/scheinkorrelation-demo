"""Einheiten des Kerns an Handrechnungen (Mini-Instanzen in conftest.py) und ihre Grundeigenschaften."""

import math

import numpy as np
import pytest

import skz_algorithm as K
import skz_constants as C
import prs_scenario as S


def _row(v):
    return np.array([v], dtype=float)


def test_correlation_regression_and_durbin_watson_by_hand(five):
    x, y = _row(five[0]), _row(five[1])
    assert K.corr_rows(x, y)[0] == pytest.approx(6 / math.sqrt(60), abs=1e-14)
    dw, r2 = K.durbin_watson_rows(x, y)
    assert dw[0] == pytest.approx(4.84 / 2.4, abs=1e-12) and r2[0] == pytest.approx(0.6, abs=1e-12)
    assert not K.warning_rows(x, y)[0]


def test_the_warning_rule_fires_when_r_squared_exceeds_dw():
    x = np.cumsum(S.VRng(1).normal(80))
    y = x + 0.3 * np.cumsum(S.VRng(2).normal(80))                     # hoher Zusammenhang, aber die Residuen sind selbst eine Irrfahrt: R² hoch, DW klein
    dw, r2 = K.durbin_watson_rows(_row(x), _row(y))
    assert r2[0] > 0.8 and dw[0] < 0.5 and K.warning_rows(_row(x), _row(y))[0]
    t = np.arange(80.0)
    assert not K.warning_rows(_row(t), _row(S.VRng(3).normal(80)))[0]    # Residuen ohne Autokorrelation: DW nahe 2


def test_lag1_autocorrelation_by_hand(five):
    assert K.lag1_rows(_row(five[0]))[0] == pytest.approx(0.4, abs=1e-14)


def test_differences_by_hand(five):
    assert K.diff_rows(_row(five[0]))[0].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert K.diff_rows(_row(five[1]))[0].tolist() == [2.0, 1.0, -1.0, 1.0]


def test_detrending_removes_a_line_exactly_and_leaves_zero_mean(five):
    line = _row([3 + 2.0 * t for t in range(8)])
    assert K.detrend_rows(line)[0] == pytest.approx(np.zeros(8), abs=1e-12)
    res = K.detrend_rows(_row(five[1]))[0]
    assert res.sum() == pytest.approx(0.0, abs=1e-12) and res == pytest.approx([-0.8, 0.6, 1.0, -0.6, -0.2], abs=1e-12)        # Residuen der Regression auf die Zeit: dieselben wie y ~ x


def test_circular_shift_p_by_hand(three):
    assert K.shift_p_rows(_row(three[0]), _row(three[1]))[0] == pytest.approx(1.0)
    line = _row(np.arange(10.0))
    assert K.shift_p_rows(line, line)[0] == pytest.approx(1 / 10)         # perfekte Gleichheit: nur die Identität (und keine andere Verschiebung) erreicht |r| = 1


def test_effective_sample_size_by_hand():
    T = 200
    x = np.cumsum(S.VRng(1).normal(T))
    prod = K.lag1_rows(_row(x))[0] ** 2
    assert K.n_eff_rows(_row(x), _row(x))[0] == pytest.approx(T * (1 - prod) / (1 + prod), abs=1e-9)
    assert K.n_eff_rows(_row(x), _row(x))[0] < 0.1 * T                        # eine Irrfahrt hat nur wenige „freie“ Beobachtungen
    white = S.VRng(2).normal((1, T))
    assert 3 <= K.n_eff_rows(white, S.VRng(3).normal((1, T)))[0] <= T


def test_effective_sample_size_is_capped_at_T_and_floored_at_three():
    T = 50
    alt = np.tile([1.0, -1.0], T // 2)[None, :] + 0.01 * S.VRng(1).normal((1, T))               # starke negative Autokorrelation: Produkt positiv (beide −1) → n_eff < T
    anti_x = alt
    anti_y = np.cumsum(S.VRng(2).normal((1, T)), axis=1)                                          # x negativ, y positiv autokorreliert: Produkt negativ
    assert K.n_eff_rows(anti_x, anti_y)[0] == T
    assert K.n_eff_rows(np.cumsum(S.VRng(4).normal((1, 500)), axis=1), np.cumsum(S.VRng(5).normal((1, 500)), axis=1))[0] >= 3


def test_test_pair_returns_all_methods_and_the_warning_rule(five):
    out = K.test_pair(*five)
    assert set(out) == set(C.METHODS) | {C.WARNING_RULE}
    r, p = out["naiv"]
    assert r == pytest.approx(0.7745966692414834) and p == pytest.approx(0.124027, abs=1e-5)         # t = r√3/√(1−r²) = 2,121 bei 3 Freiheitsgraden
    assert out["neff"][0] == pytest.approx(r)
    assert out["shift"][1] == pytest.approx(0.2)
    assert out["dw"] == (pytest.approx(2.0166666666666666), pytest.approx(0.6), False)
    assert math.isnan(out["diff"][0]) and math.isnan(out["trend"][0])                              # Differenzen von x konstant, Trend von x exakt linear: Korrelation nicht definiert


def test_the_vectorized_decisions_equal_the_p_values():
    X, Y = K.make_pair("walk", 600, 80, S.VRng(1))
    for m in C.METHODS:
        _, p = K.method_stats(m, X, Y)
        assert (K.reject_rows(m, X, Y) == (p <= C.ALPHA)).all(), m


def test_make_pair_is_reproducible_and_streams_differ():
    a = K.make_pair("ar1", 5, 30, S.VRng(1), rho=0.3, phi=0.7)
    b = K.make_pair("ar1", 5, 30, S.VRng(1), rho=0.3, phi=0.7)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all() and (K.make_pair("ar1", 5, 30, S.VRng(2))[0] != a[0]).any()


def test_a_perfect_relation_is_found_by_every_method():
    x = np.cumsum(S.VRng(1).normal((50, 104)), axis=1)
    y = x + 0.3 * S.VRng(2).normal((50, 104))
    for m in C.METHODS:
        assert K.reject_rows(m, x, y).mean() > 0.9, m
