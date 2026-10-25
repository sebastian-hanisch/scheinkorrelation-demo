"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): kleinste Fälle, inklusive Grenzen, Gleichstände, konstante Reihen, Fehlerwege."""

import math

import numpy as np
import pytest

import skz_algorithm as K
import skz_constants as C
import skz_evaluation as ev
import prs_scenario as S


def test_exactly_five_time_points_is_the_smallest_case_for_test_pair():
    x, y = S.VRng(1).normal(5), S.VRng(2).normal(5)
    out = K.test_pair(x, y)
    assert set(out) == set(C.METHODS) | {C.WARNING_RULE} and all(0 <= out[m][1] <= 1 for m in ("naiv", "neff", "shift"))


@pytest.mark.parametrize("T", [0, 1, 4])
def test_fewer_than_five_time_points_are_rejected_by_test_pair(T):
    with pytest.raises(ValueError):
        K.test_pair(np.arange(float(T)), np.arange(float(T)))


def test_unequal_lengths_are_rejected():
    with pytest.raises(ValueError):
        K.test_pair(np.arange(6.0), np.arange(7.0))


def test_make_pair_needs_three_points_and_valid_parameters():
    assert K.make_pair("walk", 2, 3, S.VRng(1))[0].shape == (2, 3)
    with pytest.raises(ValueError):
        K.make_pair("walk", 2, 2, S.VRng(1))
    with pytest.raises(ValueError):
        K.make_pair("kreis", 2, 10, S.VRng(1))
    for rho in (-0.1, 1.0):
        with pytest.raises(ValueError):
            K.make_pair("walk", 2, 10, S.VRng(1), rho=rho)
    assert K.make_pair("walk", 2, 10, S.VRng(1), rho=0.0)[0].shape == (2, 10)                  # ρ = 0 ist erlaubt


def test_ar1_phi_bounds_are_inclusive_at_zero_and_exclusive_at_one():
    e = S.VRng(1).normal((3, 10))
    assert (K.ar1_filter(e, 0.0) == e).all()                                                   # φ = 0: weißes Rauschen
    for phi in (-0.01, 1.0):
        with pytest.raises(ValueError):
            K.ar1_filter(e, phi)
    assert K.ar1_filter(e, 0.99).shape == (3, 10)


def test_constant_series_give_nan_correlation_and_no_rejection():
    c, v = np.ones((1, 30)), S.VRng(1).normal((1, 30))
    assert np.isnan(K.corr_rows(c, v)[0])
    for m in C.METHODS:
        assert not K.reject_rows(m, c, v)[0], m
    assert np.isnan(K.shift_p_rows(c, v)[0])


def test_a_perfect_correlation_gives_p_zero_and_is_rejected_by_every_method():
    x = np.cumsum(S.VRng(1).normal((1, 40)), axis=1)
    for m in ("naiv", "trend", "neff"):
        r, p = K.method_stats(m, x, 2 * x + 1)
        assert p[0] == pytest.approx(0.0, abs=1e-12) or p[0] < 1e-12, m
    assert K.reject_rows("naiv", x, 2 * x + 1)[0] and K.reject_rows("diff", x, 2 * x + 1)[0]


def test_the_shift_test_has_a_floor_of_one_over_T_and_a_ceiling_of_one():
    T = 20
    x = np.arange(float(T))[None, :]
    assert K.shift_p_rows(x, x)[0] == pytest.approx(1 / T)
    flat_like = np.tile([1.0, -1.0], T // 2)[None, :]
    assert K.shift_p_rows(flat_like, flat_like)[0] <= 1.0


def test_n_eff_stays_between_the_floor_and_T_for_a_long_random_walk():
    walk = np.cumsum(S.VRng(2).normal((1, 2000)), axis=1)
    assert 3.0 <= K.n_eff_rows(walk, walk)[0] < 100                                         # Lag-1-Produkt nahe 1: nur wenige „freie“ Beobachtungen
    c = np.tile([1.0, -1.0], 20)[None, :]
    assert K.n_eff_rows(c, walk[:, :40])[0] == 40                                           # Produkt negativ: bei T gedeckelt


def test_the_trend_slope_of_zero_gives_plain_noise():
    X, _ = K.make_pair("trend", 1, 50, S.VRng(3), slope=0.0)
    W, _ = K.make_pair("weiss", 1, 50, S.VRng(3))
    assert (X == W).all()


def test_detrending_two_points_is_exact_zero():
    assert K.detrend_rows(np.array([[1.0, 5.0]]))[0] == pytest.approx([0.0, 0.0], abs=1e-12)


def test_settings_at_all_limits_run():
    for T, rho, process in ((C.T_MIN, C.RHO_MIN, "ar1"), (C.T_MAX, C.RHO_MAX, "walk"), (C.T_MIN, C.RHO_MAX, "trend"), (C.T_MAX, C.RHO_MIN, "weiss")):
        a = ev.analyse(ev.Settings(process, C.PHI_MAX, C.SLOPE_MAX, T, rho, 0))
        assert len(a.x) == T and set(a.results) == set(C.METHODS) and all(0 <= a.results[m][1] <= 1 for m in ("naiv", "neff", "shift"))
    a = ev.analyse(ev.Settings("ar1", C.PHI_MIN, C.SLOPE_MIN, C.T_MIN, 0.0, 0))               # φ = 0 an der unteren Grenze
    assert math.isfinite(a.results["naiv"][1])


def test_every_process_runs_through_the_analysis():
    for process in C.PROCESSES:
        a = ev.analyse(ev.Settings(process, 0.9, 0.05, 80, 0.0, 4))
        assert not a.related and set(a.found) <= set(C.METHODS)


def test_the_cointegrated_pair_with_zero_noise_is_an_exact_copy():
    x, y = K.make_cointegrated_pair(3, 20, S.VRng(5), noise=0.0)
    assert (x == y).all()


def test_exact_perfect_correlation_gives_p_exactly_zero_and_constant_series_give_nan_p():
    x = np.arange(12.0)[None, :]
    for m in ("naiv", "trend", "neff"):
        r, p = K.method_stats(m, x + 0.0, x * 2.0 + 1.0)
        if m == "trend":
            assert np.isnan(p[0])                      # eine Gerade nach Abzug der Trendgeraden: Korrelation nicht definiert
        else:
            assert r[0] == 1.0 and p[0] == 0.0, m
    r, p = K.method_stats("naiv", np.ones((1, 12)), x)
    assert np.isnan(r[0]) and np.isnan(p[0])
    r, p = K.method_stats("naiv", x, -x)
    assert r[0] == -1.0 and p[0] == 0.0                 # auch r = −1 (|t| unendlich) gibt p = 0


def test_the_cointegrated_pair_adds_the_noise_with_a_plus_sign():
    rng = S.VRng(9)
    x_expected = np.cumsum(rng.normal((3, 20)), axis=1)
    eps = rng.normal((3, 20))
    x, y = K.make_cointegrated_pair(3, 20, S.VRng(9), noise=0.7)
    assert (x == x_expected).all() and y == pytest.approx(x_expected + 0.7 * eps, abs=1e-15)
