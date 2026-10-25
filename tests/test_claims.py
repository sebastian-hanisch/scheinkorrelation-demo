"""Jede Zahl der README steht hier und wird aus dem vorgerechneten Raster (data/spurious_grid.json) oder über dieselben Auswertungsfunktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4).

Zufall nur aus SplitMix64 mit festen Seeds. Raster-Werte stehen in der README als Prozent mit einer Nachkommastelle; die Tabellen werden Zeile für Zeile in der Rundung der App (`_p`) gegen die
README geprüft, sonst gilt eine Toleranz von einem halben Zehntelprozent. Live nachgerechnete Werte (Varianz, Verteilung von r) lassen drei Nachkommastellen zu."""

import re
from pathlib import Path

import numpy as np
import pytest

import skz_algorithm as K
import skz_constants as C
import skz_evaluation as ev
import prs_scenario as S

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
METHODS = list(C.METHODS) + ["dw"]
TOL = 0.0006


def _p(v):
    return f"{100 * v:.1f}".replace(".", ",")


def _d(v, digits=3):
    return f"{v:.{digits}f}".replace(".", ",")


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None
    return g


def _fa(grid, key, T):
    return ev.find(grid["fa"], process=key, T=T)


def test_claim_the_naive_false_alarm_table_is_the_grid_in_the_apps_rounding(grid):
    """README: Tabelle „Fehlalarm des naiven Tests“ (sechs Prozesse, fünf Längen)."""
    for key in C.GRID_PROCESSES:
        row = f"| {C.GRID_NAMES[key]} | " + " | ".join(_p(_fa(grid, key, T)["rate"]["naiv"]) for T in C.GRID_TS) + " |"
        assert row in README, row


def test_claim_the_remedy_table_at_104_is_the_grid_in_the_apps_rounding(grid):
    """README: Tabelle „Gegenmittel bei unabhängigen Reihen, T = 104“."""
    for key in C.GRID_PROCESSES:
        r = _fa(grid, key, 104)["rate"]
        row = f"| {C.GRID_NAMES[key]} | " + " | ".join(_p(r[m]) for m in METHODS) + " |"
        assert row in README, row


def test_claim_the_relation_table_is_the_grid_in_the_apps_rounding(grid):
    """README: Tabelle „Echter Zusammenhang, T = 104“ (alle zehn Zeilen)."""
    labels = {"walk": "zwei Irrfahrten, Zuwächse korreliert", "ar1": "zwei AR(1) 0,9, Innovationen korreliert", "trend": "gemeinsamer Trend, Rauschen korreliert"}
    for sc, label in labels.items():
        for i, rho in enumerate(C.REL_RHOS):
            r = ev.find(grid["rel"], scenario=sc, rho=rho, T=104)["rate"]
            rho_txt = "0" if rho == 0 else _d(rho, 1)
            lead = f"| {label} |" if i == 0 else "| |"
            row = f"{lead} {rho_txt} | " + " | ".join(_p(r[m]) for m in METHODS) + " |"
            assert row in README, row
    c = ev.find(grid["rel"], scenario="coint", rho=0.0, T=104)["rate"]
    assert "| gemeinsame Irrfahrt, y = x + Rauschen | – | " + " | ".join(_p(c[m]) for m in METHODS) + " |" in README


def test_claim_random_walk_false_alarm_grows_and_white_noise_stays(grid):
    """README: Irrfahrt 52,5 / 67,7 / 76,1 / 86,5 / 89,4 %; weißes Rauschen 4,8 bis 5,0 %; AR(1) 0,9 bis 53,0; AR(1) 0,99 bis 83,0; Trend 8,9 % bei T = 26, 100,0 % ab T = 104."""
    walk = [_fa(grid, "walk", T)["rate"]["naiv"] for T in C.GRID_TS]
    assert walk == pytest.approx([0.525, 0.677, 0.761, 0.865, 0.894], abs=TOL) and walk == sorted(walk)
    white = [_fa(grid, "weiss", T)["rate"]["naiv"] for T in C.GRID_TS]
    assert (min(white), max(white)) == pytest.approx((0.048, 0.050), abs=TOL)
    assert _fa(grid, "ar09", 520)["rate"]["naiv"] == pytest.approx(0.530, abs=TOL) and _fa(grid, "ar099", 520)["rate"]["naiv"] == pytest.approx(0.830, abs=TOL)
    assert [_fa(grid, "trend", T)["rate"]["naiv"] for T in C.GRID_TS] == pytest.approx([0.089, 0.793, 1.0, 1.0, 1.0], abs=TOL)


def test_claim_r_does_not_shrink_for_walks(grid):
    """README: Streuung von r (Irrfahrt) 0,489 / 0,493 / 0,488 / 0,484 / 0,484, mittleres |r| 0,417 bis 0,424, |r| > 0,5 in 38,4 bis 40,4 %; weißes Rauschen 0,200 / 0,139 / 0,099 / 0,062 / 0,044."""
    walk = [_fa(grid, "walk", T)["r"] for T in C.GRID_TS]
    assert [r["sd"] for r in walk] == pytest.approx([0.489, 0.493, 0.488, 0.484, 0.484], abs=TOL)
    assert (min(r["mean_abs"] for r in walk), max(r["mean_abs"] for r in walk)) == pytest.approx((0.417, 0.424), abs=TOL)
    assert (min(r["share_big"] for r in walk), max(r["share_big"] for r in walk)) == pytest.approx((0.384, 0.404), abs=TOL)
    assert [_fa(grid, "weiss", T)["r"]["sd"] for T in C.GRID_TS] == pytest.approx([0.200, 0.139, 0.099, 0.062, 0.044], abs=TOL)


def test_claim_live_histogram_shares():
    """README: live, 2 000 Paare, T = 104: Irrfahrt 76,7 % über der Schwelle 0,193, weißes Rauschen 5,4 %."""
    crit = ev.r_critical(104)
    assert crit == pytest.approx(0.193, abs=TOL)
    assert (np.abs(ev.r_sample("walk", 104)) > crit).mean() == pytest.approx(0.767, abs=TOL) and (np.abs(ev.r_sample("weiss", 104)) > crit).mean() == pytest.approx(0.054, abs=TOL)


def test_claim_differencing(grid):
    """README: Differenzieren bei Irrfahrt/AR(1) ≥ 0,9: 4,4 bis 5,1 % (T = 104), 4,4 bis 5,6 % (alle T); weißes Rauschen 10,4 bis 11,8 %; Trend 10,2 bis 10,8 %; AR(1) 0,5: 6,7 bis 7,8 %."""
    diff = lambda key, T: _fa(grid, key, T)["rate"]["diff"]          # noqa: E731
    healed = [diff(k, 104) for k in ("walk", "ar09", "ar099")]
    assert (min(healed), max(healed)) == pytest.approx((0.044, 0.051), abs=TOL)
    every = [diff(k, T) for k in ("walk", "ar09", "ar099") for T in C.GRID_TS]
    assert (min(every), max(every)) == pytest.approx((0.044, 0.056), abs=TOL)
    white = [diff("weiss", T) for T in C.GRID_TS]
    trend = [diff("trend", T) for T in C.GRID_TS]
    ar05 = [diff("ar05", T) for T in C.GRID_TS]
    assert (min(white), max(white)) == pytest.approx((0.104, 0.118), abs=TOL) and (min(trend), max(trend)) == pytest.approx((0.102, 0.108), abs=TOL) and (min(ar05), max(ar05)) == pytest.approx((0.067, 0.078), abs=TOL)


def test_claim_detrending_n_eff_and_shift(grid):
    """README: Trend weg: Trend 5,0 %, Irrfahrt 64,8 % (T = 104) und 84,2 % (T = 520), AR(1) 0,9 47,3 %; n_eff 4,0 bis 5,1 % (T = 104; Rauschen, AR(1), Irrfahrt), Trend 100,0 %; Verschiebung 4,9 bis 5,5 % (Rauschen, AR(1) 0,5 und 0,9), 16,0 / 22,0 / 99,2 %."""
    r = {k: _fa(grid, k, 104)["rate"] for k in C.GRID_PROCESSES}
    assert (r["trend"]["trend"], r["walk"]["trend"], r["ar09"]["trend"]) == pytest.approx((0.050, 0.648, 0.473), abs=TOL) and _fa(grid, "walk", 520)["rate"]["trend"] == pytest.approx(0.842, abs=TOL)
    neff = [r[k]["neff"] for k in ("weiss", "ar05", "ar09", "ar099", "walk")]
    assert (min(neff), max(neff)) == pytest.approx((0.040, 0.051), abs=TOL) and r["trend"]["neff"] == 1.0
    shift = [r[k]["shift"] for k in ("weiss", "ar05", "ar09")]
    assert (min(shift), max(shift)) == pytest.approx((0.049, 0.055), abs=TOL) and (r["ar099"]["shift"], r["walk"]["shift"], r["trend"]["shift"]) == pytest.approx((0.160, 0.220, 0.992), abs=TOL)


def test_claim_warning_rule(grid):
    """README: Warnregel bei unabhängigen Irrfahrten 10,3 / 34,3 / 55,9 / 74,1 / 81,5 % (T = 26 bis 520); Trend, weißes Rauschen und gemeinsame Irrfahrt 0,0 %."""
    assert [_fa(grid, "walk", T)["rate"]["dw"] for T in C.GRID_TS] == pytest.approx([0.103, 0.343, 0.559, 0.741, 0.815], abs=TOL)
    assert all(_fa(grid, k, T)["rate"]["dw"] == 0.0 for k in ("trend", "weiss") for T in C.GRID_TS)
    assert all(ev.find(grid["rel"], scenario="coint", rho=0.0, T=T)["rate"]["dw"] == 0.0 for T in C.REL_TS)


def test_claim_relations(grid):
    """README: Irrfahrten ohne Zusammenhang 76,1 %, mit ρ = 0,3 79,4 % (naiv), Differenzieren 5,2 gegen 88,2 %, n_eff 8,6 %; AR(1) 0,9: Differenzieren 5,2 gegen 87,9 %, n_eff 20,4 %; gemeinsamer Trend 100,0 in beiden Fällen; gemeinsame Irrfahrt 98,0 bis 100,0 %."""
    cell = lambda sc, rho: ev.find(grid["rel"], scenario=sc, rho=rho, T=104)["rate"]          # noqa: E731
    assert (cell("walk", 0.0)["naiv"], cell("walk", 0.3)["naiv"], cell("walk", 0.0)["diff"], cell("walk", 0.3)["diff"], cell("walk", 0.3)["neff"]) == pytest.approx((0.761, 0.794, 0.052, 0.882, 0.086), abs=TOL)
    assert (cell("ar1", 0.0)["diff"], cell("ar1", 0.3)["diff"], cell("ar1", 0.3)["neff"]) == pytest.approx((0.052, 0.879, 0.204), abs=TOL)
    assert cell("trend", 0.0)["naiv"] == 1.0 == cell("trend", 0.3)["naiv"]
    coint = cell("coint", 0.0)
    assert (min(coint[m] for m in C.METHODS), max(coint[m] for m in C.METHODS)) == pytest.approx((0.980, 1.0), abs=TOL)


def test_claim_variance_live():
    """README: T · Var(r) für T = 104 / 520: φ = 0: 1,02 / 0,98; φ = 0,5: 1,60 / 1,65; φ = 0,9: 7,80 / 9,34; φ = 0,99: 20,59 / 58,55; Formel 1,00 / 1,67 / 9,53 / 99,50."""
    expected = {0.0: (1.016, 0.978, 1.0), 0.5: (1.598, 1.645, 1.667), 0.9: (7.796, 9.335, 9.526), 0.99: (20.590, 58.549, 99.503)}
    for phi, (a, b, f) in expected.items():
        cells = [ev.variance_cell(phi, T) for T in C.VAR_TS]
        assert [c["t_var"] for c in cells] == pytest.approx([a, b], abs=0.0006) and cells[0]["formula"] == pytest.approx(f, abs=0.0006)
    assert ev.variance_cell(0.99, 104)["n_eff"] == pytest.approx(104 / 99.503, abs=0.001)          # in der App „1.0“


def test_claim_hand_values(five, three):
    """README: r = 6/√60 = 0,7746, DW = 4,84/2,4 = 2,0167, R² = 0,6 (keine Warnung); Lag-1-Autokorrelation 0,4; Differenzen (1, 1, 1, 1); Verschiebung p = 3/3."""
    x, y = np.array([five[0]]), np.array([five[1]])
    assert K.corr_rows(x, y)[0] == pytest.approx(0.7746, abs=TOL / 6) and K.lag1_rows(x)[0] == pytest.approx(0.4)
    dw, r2 = K.durbin_watson_rows(x, y)
    assert dw[0] == pytest.approx(2.0167, abs=1e-4) and r2[0] == pytest.approx(0.6) and not K.warning_rows(x, y)[0]
    assert K.diff_rows(x)[0].tolist() == [1.0, 1.0, 1.0, 1.0] and K.shift_p_rows(np.array([three[0]]), np.array([three[1]]))[0] == pytest.approx(1.0)


def test_claim_seed_spread_is_at_most_2_6_points(grid):
    """README: die Streuung über die Seeds liegt in jeder Zelle bei höchstens 2,6 Punkten."""
    assert max(max(c["sd"].values()) for c in grid["fa"] + grid["rel"]) == pytest.approx(0.026, abs=TOL)


def test_claim_preset_help_numbers(grid):
    """Hilfetexte der Presets: jede Prozentzahl steht im Raster (Zelle und Verfahren des Presets) in der Rundung der App."""
    fa = lambda key, m: _fa(grid, key, 104)["rate"][m]                       # noqa: E731
    rel = lambda rho, m: ev.find(grid["rel"], scenario="walk", rho=rho, T=104)["rate"][m]          # noqa: E731
    expected = {
        "Standardfall (Voreinstellung)": [fa("walk", "naiv"), fa("walk", "diff"), fa("walk", "neff")],
        "Weißes Rauschen (Kontrolle)": [fa("weiss", "naiv"), fa("weiss", "diff")],
        "AR(1) mit φ = 0,9": [fa("ar09", "naiv"), fa("ar09", "diff"), fa("ar09", "neff"), fa("ar09", "shift")],
        "Trend und Rauschen": [fa("trend", "naiv"), fa("trend", "neff"), fa("trend", "trend")],
        "Echter Zusammenhang (Irrfahrten, ρ = 0,3)": [rel(0.3, "naiv"), rel(0.0, "naiv"), rel(0.3, "diff"), rel(0.3, "neff")],
        "Überdifferenziert (weißes Rauschen, 260 Wochen)": [_fa(grid, "weiss", 260)["rate"]["naiv"], _fa(grid, "weiss", 260)["rate"]["diff"]],
    }
    assert set(expected) == set(C.PRESETS)
    for name, values in expected.items():
        text = C.PRESET_HELP[name]
        mentioned = [m.replace(",", ".") for m in re.findall(r"(\d+,\d+) %", text)]
        assert mentioned == [_p(v).replace(",", ".") for v in values], (name, mentioned)
    assert (C.PRESETS["Überdifferenziert (weißes Rauschen, 260 Wochen)"]["T"], C.PRESETS["Standardfall (Voreinstellung)"]["T"]) == (260, 104)


def test_claim_readme_mentions_its_key_numbers():
    """Verbindet die README mit den Tests: die Zahlen, die dort als Kernbefunde stehen, kommen im Text vor."""
    for needle in ("76,1 %", "89,4 %", "0,489", "0,484", "11,3 %", "10,3 %", "64,8 %", "100,0 %", "8,6 % statt 88,2 %", "81,5 %", "3 Sekunden", "höchstens 2,6 Punkten", "76,7 %", "58,55"):
        assert needle in README, needle


def test_claim_grid_build_time(grid):
    """README: 3 Sekunden auf 10 Prozessen."""
    assert grid["meta"]["seconds"] == 3


def test_claim_stream_is_reproducible():
    X, Y = K.make_pair("walk", 3, 20, S.VRng(1))
    assert X[0, 0] == pytest.approx(K.make_pair("walk", 3, 20, S.VRng(1))[0][0, 0])
