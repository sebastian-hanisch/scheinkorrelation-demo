"""Eigenschaften der Verfahren: Invarianz, Zweige (jedes Gegenmittel hilft irgendwo und scheitert irgendwo), die Warnregel und das Verhalten von r bei Irrfahrten.

Einzel-Seed-Tests auf „Test schlägt (nicht) an“ sind bei solchen Prozessen fragil (Lehre aus Stück 3): hier zählen Anteile über viele Läufe mit Schwellen."""

import numpy as np
import pytest

import skz_algorithm as K
import skz_constants as C
import prs_scenario as S

RUNS = 600
T = 104


def _rates(process, seed, rho=0.0, **kw):
    X, Y = K.make_pair(process, RUNS, T, S.VRng(seed), rho=rho, **kw)
    return {m: float(K.reject_rows(m, X, Y).mean()) for m in C.METHODS}, float(K.warning_rows(X, Y).mean())


def test_every_remedy_helps_somewhere_and_fails_somewhere():
    rates = {p: _rates(p, 10 + i)[0] for i, p in enumerate(("weiss", "ar1", "walk", "trend"))}
    naive_fails = {p for p in rates if rates[p]["naiv"] > 0.15}
    assert naive_fails == {"ar1", "walk", "trend"}
    for m in ("diff", "trend", "neff", "shift"):
        helps = {p for p in naive_fails if rates[p][m] < 0.09}
        fails = {p for p in rates if rates[p][m] > 0.09}
        assert helps and fails, (m, helps, fails)


def test_the_specific_helpers_and_failures():
    r = {p: _rates(p, 20 + i)[0] for i, p in enumerate(("weiss", "ar1", "walk", "trend"))}
    assert r["walk"]["diff"] < 0.09 and r["weiss"]["diff"] > 0.08 and r["trend"]["diff"] > 0.08          # Differenzieren: heilt die Irrfahrt, schadet bei Rauschen und Trend
    assert r["trend"]["trend"] < 0.09 and r["walk"]["trend"] > 0.5                                         # Trendbereinigung: nur der Trend
    assert r["walk"]["neff"] < 0.09 and r["trend"]["neff"] > 0.9                                           # n_eff: geht bei der Irrfahrt, nicht beim Trend
    assert r["ar1"]["shift"] < 0.09 and r["walk"]["shift"] > 0.12 and r["trend"]["shift"] > 0.9            # Verschiebung: nur bei kurzem Gedächtnis


def test_the_warning_rule_fires_more_often_for_independent_walks_than_for_white_noise():
    walk = _rates("walk", 30)[1]
    white = _rates("weiss", 31)[1]
    assert walk > 0.3 and white == 0.0


def test_the_naive_false_alarm_of_the_walk_does_not_shrink_with_T():
    short = np.mean([K.reject_rows("naiv", *K.make_pair("walk", RUNS, 26, S.VRng(40))).mean()])
    long = np.mean([K.reject_rows("naiv", *K.make_pair("walk", RUNS, 400, S.VRng(41))).mean()])
    assert long > short + 0.2


def test_the_spread_of_r_between_walks_is_stable_in_T_but_shrinks_for_white_noise():
    sd = lambda p, t, seed: K.corr_rows(*K.make_pair(p, 1500, t, S.VRng(seed))).std()               # noqa: E731
    assert abs(sd("walk", 30, 50) - sd("walk", 300, 51)) < 0.05
    assert sd("weiss", 300, 52) < 0.5 * sd("weiss", 30, 53)


@pytest.mark.parametrize("seed", range(5))
def test_correlation_tests_are_invariant_to_shifting_and_scaling_both_series(seed):
    X, Y = K.make_pair("walk", 8, 60, S.VRng(seed))
    for m in C.METHODS:
        a = K.method_stats(m, X, Y)
        b = K.method_stats(m, 3.0 * X + 5.0, 0.2 * Y - 1.0)
        assert b[0] == pytest.approx(a[0], abs=1e-9) and b[1] == pytest.approx(a[1], abs=1e-9), m
    assert K.durbin_watson_rows(X, Y)[0] == pytest.approx(K.durbin_watson_rows(3.0 * X + 5.0, 0.2 * Y - 1.0)[0], abs=1e-9)


@pytest.mark.parametrize("seed", range(4))
def test_the_shift_test_is_symmetric_and_invariant_to_rolling_both_series(seed):
    X, Y = K.make_pair("ar1", 6, 50, S.VRng(seed), rho=0.3)
    base = K.shift_p_rows(X, Y)
    assert K.shift_p_rows(Y, X) == pytest.approx(base, abs=1e-12)                                       # dieselben Verschiebungen, andere Richtung
    assert K.shift_p_rows(np.roll(X, 7, axis=1), np.roll(Y, 7, axis=1)) == pytest.approx(base, abs=1e-12)   # beide gleich verschoben: dieselbe Ausrichtung


def test_the_detrended_test_ignores_a_common_linear_trend():
    X, Y = K.make_pair("weiss", 20, 80, S.VRng(5), rho=0.0)
    t = np.arange(80)[None, :]
    a = K.method_stats("trend", X, Y)
    b = K.method_stats("trend", X + 0.3 * t, Y + 0.7 * t + 2)
    assert b[0] == pytest.approx(a[0], abs=1e-9) and b[1] == pytest.approx(a[1], abs=1e-9)


def test_differencing_ignores_a_linear_trend_and_the_level():
    X, Y = K.make_pair("weiss", 20, 80, S.VRng(6))
    t = np.arange(80)[None, :]
    a = K.method_stats("diff", X, Y)
    b = K.method_stats("diff", X + 0.3 * t + 9, Y + 0.7 * t)
    assert b[0] == pytest.approx(a[0], abs=1e-9) and b[1] == pytest.approx(a[1], abs=1e-9)


def test_a_true_relation_gets_stronger_with_rho_for_the_difference_test():
    low = _rates("walk", 60, rho=0.1)[0]["diff"]
    mid = _rates("walk", 61, rho=0.3)[0]["diff"]
    high = _rates("walk", 62, rho=0.6)[0]["diff"]
    assert low < mid < high and high > 0.95
