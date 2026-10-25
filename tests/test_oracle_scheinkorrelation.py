"""Orakel (ORACLE-PLAYBOOK): jedes Verfahren gegen einen anderen Rechenweg.

* Pearson und p-Werte: scipy.stats (`pearsonr`, `t.sf`); Differenzen gegen `numpy.diff`; Trendbereinigung gegen `numpy.polyfit`.
* Durbin–Watson, Lag-1-Autokorrelation, Verschiebungstest: Schleifen über die Definitionen (`numpy.roll` für die Verschiebungen).
* Prozesse: Autokorrelation des AR(1) gegen φ^k, Var(x_t) = t der Irrfahrt, die Varianz von r bei unabhängigen AR(1)-Paaren gegen die geschlossene Formel (1 + φ²)/((1 − φ²) T).
Die Bibliotheken stehen nur im Test-venv; das Orakel braucht sie nicht zur Laufzeit. `statsmodels` fehlt dort (kein Orakel für DW und ACF außer der Schleife)."""

import math

import numpy as np
import pytest

import skz_algorithm as K
import prs_scenario as S

scipy_stats = pytest.importorskip("scipy.stats")


def _pair(seed, T=40, process="walk", rho=0.0):
    X, Y = K.make_pair(process, 1, T, S.VRng(seed), rho=rho)
    return X[0], Y[0]


def _p_t(r, df):
    t = abs(r) * math.sqrt(df / (1 - r * r))
    return 2.0 * scipy_stats.t.sf(t, df)


@pytest.mark.parametrize("seed", range(8))
def test_naive_and_difference_tests_match_scipy(seed):
    x, y = _pair(seed)
    T = len(x)
    out = K.test_pair(x, y)
    r, p = scipy_stats.pearsonr(x, y)
    assert out["naiv"][0] == pytest.approx(r, abs=1e-12) and out["naiv"][1] == pytest.approx(p, abs=1e-10)
    rd, pd = scipy_stats.pearsonr(np.diff(x), np.diff(y))
    assert out["diff"][0] == pytest.approx(rd, abs=1e-12) and out["diff"][1] == pytest.approx(pd, abs=1e-10)           # T − 1 Beobachtungen, T − 3 Freiheitsgrade
    assert T - 3 == 37


@pytest.mark.parametrize("seed", range(8))
def test_detrending_matches_polyfit_and_the_test_uses_t_minus_two_degrees(seed):
    x, y = _pair(seed, process="trend")
    T = len(x)
    t = np.arange(T)
    rx = x - np.polyval(np.polyfit(t, x, 1), t)
    ry = y - np.polyval(np.polyfit(t, y, 1), t)
    r = scipy_stats.pearsonr(rx, ry)[0]
    out = K.test_pair(x, y)["trend"]
    assert out[0] == pytest.approx(r, abs=1e-10) and out[1] == pytest.approx(_p_t(r, T - 2), abs=1e-10)


@pytest.mark.parametrize("seed", range(6))
def test_effective_sample_size_test_matches_the_definition(seed):
    x, y = _pair(seed, T=60)
    T = len(x)

    def lag1(v):
        c = v - v.mean()
        return sum(c[i] * c[i + 1] for i in range(T - 1)) / sum(c * c)

    prod = lag1(x) * lag1(y)
    n_eff = min(T, max(3.0, T * (1 - prod) / (1 + prod)))
    r = scipy_stats.pearsonr(x, y)[0]
    out = K.test_pair(x, y)["neff"]
    assert out[0] == pytest.approx(r, abs=1e-12) and out[1] == pytest.approx(_p_t(r, n_eff - 2), abs=1e-10)


@pytest.mark.parametrize("seed", range(6))
def test_circular_shift_test_matches_the_loop_over_np_roll(seed):
    x, y = _pair(seed, T=30, process="ar1", rho=0.2)
    T = len(x)
    rs = [scipy_stats.pearsonr(x, np.roll(y, s))[0] for s in range(T)]
    expected = sum(abs(v) >= abs(rs[0]) - 1e-12 for v in rs) / T
    assert K.test_pair(x, y)["shift"][1] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_durbin_watson_and_r_squared_match_the_definition(seed):
    x, y = _pair(seed, T=50)
    T = len(x)
    xc, yc = x - x.mean(), y - y.mean()
    b = sum(xc * yc) / sum(xc * xc)
    u = [yc[i] - b * xc[i] for i in range(T)]
    dw = sum((u[i] - u[i - 1]) ** 2 for i in range(1, T)) / sum(v * v for v in u)
    got = K.test_pair(x, y)["dw"]
    assert got[0] == pytest.approx(dw, abs=1e-12) and got[1] == pytest.approx(scipy_stats.pearsonr(x, y)[0] ** 2, abs=1e-12) and got[2] == (got[1] > got[0])


def test_ar1_autocorrelation_decays_with_phi_to_the_k():
    phi = 0.8
    X = K.ar1_filter(S.VRng(1).normal((200, 4000)), phi)           # lange Reihen: die Verzerrung der Stichproben-ACF (Mittelwertabzug) schrumpft mit 1/T
    Xc = X - X.mean(1, keepdims=True)
    for k in (1, 2, 5):
        acf = ((Xc[:, k:] * Xc[:, :-k]).sum(1) / (Xc * Xc).sum(1)).mean()
        assert acf == pytest.approx(phi**k, abs=0.03), k


def test_ar1_variance_is_stationary_from_the_first_value():
    X = K.ar1_filter(S.VRng(2).normal((20000, 5)), 0.9)
    assert X.var(0) == pytest.approx(np.full(5, 1 / (1 - 0.81)), rel=0.05)


def test_random_walk_variance_grows_with_time():
    X = K.process_from_innovations("walk", S.VRng(3).normal((20000, 10)))
    assert X.var(0) == pytest.approx(np.arange(1.0, 11.0), rel=0.05)


@pytest.mark.parametrize("phi,tol", [(0.0, 0.06), (0.5, 0.08), (0.9, 0.14)])
def test_variance_of_r_between_independent_ar1_series_matches_the_formula(phi, tol):
    """T · Var(r) → (1 + φ²)/(1 − φ²) für T ≫ 1/(1 − φ); 2 000 Paare der Länge 2 000 (Monte-Carlo-Fehler der Varianz etwa 3 %)."""
    T = 2000
    X, Y = K.make_pair("ar1", 2000, T, S.VRng(5), phi=phi)
    r = K.corr_rows(X, Y)
    assert T * r.var() == pytest.approx((1 + phi**2) / (1 - phi**2), rel=tol)


def test_the_innovation_correlation_is_rho_for_white_noise_pairs():
    X, Y = K.make_pair("weiss", 1, 200_000, S.VRng(7), rho=0.4)
    assert K.corr_rows(X, Y)[0] == pytest.approx(0.4, abs=0.01)
    X, Y = K.make_pair("weiss", 1, 200_000, S.VRng(7), rho=0.0)
    assert abs(K.corr_rows(X, Y)[0]) < 0.01


def test_the_cointegrated_pair_differs_by_white_noise_of_the_given_size():
    x, y = K.make_cointegrated_pair(300, 200, S.VRng(9), noise=0.7)
    d = y - x
    assert d.std() == pytest.approx(0.7, rel=0.02) and abs(d.mean()) < 0.02 and np.abs(np.diff(d, axis=1)).mean() > 0.5       # die Differenz hat keine Irrfahrt-Struktur
