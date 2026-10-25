"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen, nicht Seeds. Damit prüft der Test die Grenzen (> 20 %, ≤ 7,5 %, > 10 %) und jede genannte Zahl direkt."""

import numpy as np

import skz_constants as C
import skz_evaluation as ev

M = list(C.METHODS)


def analysis(process="walk", found=(), related=False, T=104, rho=0.0, warning=False):
    results = {m: (0.4321, 0.0123 if m in found else 0.4567) for m in M}
    return ev.Analysis(ev.Settings(process, 0.9, 0.05, T, rho, 1), (0.0,), (0.0,), results, tuple(found), 0.456, 0.7777, warning, related)


def cell(process="walk", T=104, runs=4000, **rates):
    rate = dict.fromkeys(ev.RATE_KEYS, 0.05)
    rate.update(rates)
    return {"process": process, "T": T, "runs": runs, "rate": rate, "r": {"sd": 0.4876, "mean_abs": 0.4179, "q05": -0.769, "q95": 0.771, "share_big": 0.394}}


# --- Hauptansicht -------------------------------------------------------------------------------------------------------------------------

def test_main_unrelated_without_any_finder_is_a_success():
    kind, text = ev.verdict_main(analysis())
    assert kind == "success" and "kein Verfahren schlägt an" in text and "r = **0.43**" in text and "naiver p-Wert **0.457**" in text and "(Irrfahrt, T = 104 Wochen)" in text
    assert "R² = 0.78, DW = 0.46, **keine Warnung**" in text


def test_main_unrelated_with_only_the_naive_test_is_a_false_alarm_warning():
    kind, text = ev.verdict_main(analysis(found=("naiv",), warning=True))
    assert kind == "warning" and "der naive Test schlägt an, ein Fehlalarm, die Gegenmittel nicht" in text and "**Warnung**" in text


def test_main_unrelated_with_several_finders_names_them_all():
    kind, text = ev.verdict_main(analysis(found=("naiv", "neff")))
    assert kind == "warning" and "**naiver Test, n_eff**" in text and "alles Fehlalarme" in text
    kind, text = ev.verdict_main(analysis(found=("diff",)))
    assert kind == "warning" and "**Differenzieren**" in text and "der naive Test schlägt an" not in text


def test_main_related_without_a_finder_is_info():
    kind, text = ev.verdict_main(analysis(related=True, rho=0.3))
    assert kind == "info" and "(ρ = 0.3)" in text and "kein Verfahren findet ihn" in text


def test_main_related_names_finders_and_missers():
    kind, text = ev.verdict_main(analysis(related=True, rho=0.3, found=("diff", "trend")))
    assert kind == "success" and "es finden ihn: **Differenzieren, Trend weg**" in text and "es übersehen ihn: naiver Test, n_eff, Verschiebung" in text
    kind, text = ev.verdict_main(analysis(related=True, rho=0.3, found=tuple(M)))
    assert kind == "success" and "übersehen" not in text


# --- E1, E2 -------------------------------------------------------------------------------------------------------------------------------

def test_fa_verdict_warns_above_twenty_percent_exclusive():
    table = {"walk": cell(naiv=0.768), "weiss": cell("weiss", naiv=0.051)}
    kind, text = ev.verdict_fa(table, 104)
    assert kind == "warning" and "Bei T = 104 Wochen" in text and "**76.8 %** der 4000 Läufe" in text and "bei weißem Rauschen in 5.1 %" in text
    assert ev.verdict_fa({"walk": cell(naiv=0.2), "weiss": cell("weiss")}, 26)[0] == "info"
    assert ev.verdict_fa({"walk": cell(naiv=0.2001), "weiss": cell("weiss")}, 26)[0] == "warning"


def test_r_verdict_reports_spread_and_share():
    kind, text = ev.verdict_r(cell("walk", 104))
    assert kind == "info" and "Irrfahrt, T = 104" in text and "**0.488**" in text and "1/√T = 0.098" in text and "0.418" in text and "**39.4 %**" in text


def test_hist_verdict_threshold_is_exclusive_at_ten_percent():
    crit = ev.r_critical(104)
    base = np.zeros(100)
    s = base.copy()
    s[:10] = crit + 0.1
    kind, text = ev.verdict_hist(s, 104)
    assert kind == "success" and "**10.0 %**" in text and f"{crit:.3f}" in text and "100 gezogenen Paaren" in text
    s[:11] = crit + 0.1
    kind, text = ev.verdict_hist(s, 104)
    assert kind == "warning" and "**11.0 %**" in text
    s2 = base.copy()
    s2[:3] = -(crit + 0.1)                                       # negative r zählen mit
    assert "**3.0 %**" in ev.verdict_hist(s2, 104)[1]


# --- E3 -------------------------------------------------------------------------------------------------------------------------------------

def _table(**per_process):
    return {key: cell(key, 104, **per_process.get(key, {})) for key in C.GRID_PROCESSES}


def test_methods_verdict_counts_the_processes_a_method_holds():
    table = _table(weiss={"diff": 0.103, "naiv": 0.05}, trend={"diff": 0.107, "neff": 1.0, "naiv": 1.0, "shift": 0.997, "trend": 0.049},
                   walk={"naiv": 0.768, "trend": 0.661, "shift": 0.222}, ar099={"shift": 0.161, "naiv": 0.731, "trend": 0.638}, ar09={"naiv": 0.51, "trend": 0.486}, ar05={"naiv": 0.124, "trend": 0.125})
    kind, text = ev.verdict_methods(table, 104)
    assert kind == "warning" and "Bei T = 104 hält (Fehlalarm höchstens 7.5 %)" in text and "**Kein Verfahren hält in allen.**" in text
    assert "naiver Test in **1 von 6**" in text and "Differenzieren in **4 von 6**" in text and "Trend weg in **2 von 6**" in text and "n_eff in **5 von 6**" in text and "Verschiebung in **3 von 6**" in text


def test_methods_verdict_threshold_is_inclusive_and_all_hold_is_a_success():
    kind, text = ev.verdict_methods(_table(), 52)
    assert kind == "success" and "In allen: naiver Test, Differenzieren, Trend weg, n_eff, Verschiebung." in text and "**5 von 6**" not in text
    assert "naiver Test in **6 von 6**" in text
    edge = _table(walk={"diff": 0.075})
    assert "Differenzieren in **6 von 6**" in ev.verdict_methods(edge, 52)[1]
    over = _table(walk={"diff": 0.0751})
    assert "Differenzieren in **5 von 6**" in ev.verdict_methods(over, 52)[1]


def test_overdiff_verdict_names_the_three_rates():
    table = _table(walk={"diff": 0.044}, weiss={"diff": 0.113}, trend={"diff": 0.103})
    kind, text = ev.verdict_overdiff(table, 104)
    assert kind == "info" and "Irrfahrt (4.4 %)" in text and "weißem Rauschen auf **11.3 %**" in text and "Trend plus Rauschen auf **10.3 %**" in text and "(T = 104)" in text


def test_warning_verdict_reads_the_walk_series_and_the_trend_at_the_longest_T():
    fa = [cell("walk", T, dw=v) for T, v in zip(C.GRID_TS, (0.101, 0.333, 0.557, 0.73, 0.828))] + [cell("trend", 520, dw=0.0)]
    kind, text = ev.verdict_warning({"fa": fa[::-1]})
    assert kind == "info" and "**10.1 %** der Läufe bei T = 26" in text and "**82.8 %** bei T = 520" in text and "bei Trend plus Rauschen warnt sie in 0.0 %" in text


# --- E4, E5 ---------------------------------------------------------------------------------------------------------------------------

def _rel(scenario, rho, T, **rates):
    rate = dict.fromkeys(ev.RATE_KEYS, 0.05)
    rate.update(rates)
    return {"scenario": scenario, "rho": rho, "T": T, "runs": 4000, "rate": rate}


def test_relation_verdict_names_naive_and_best_method():
    rows = [_rel("walk", 0.0, 104, naiv=0.762), _rel("walk", 0.3, 104, naiv=0.797, diff=0.878, neff=0.077, trend=0.737)]
    kind, text = ev.verdict_relation(rows, "walk", 104)
    assert kind == "info" and "zwei Irrfahrten, Zuwächse korreliert, T = 104" in text and "ρ = 0) verwirft der naive Test in **76.2 %**" in text and "mit ρ = 0,3 in **79.7 %**" in text
    assert "Differenzieren (**87.8 %**)" in text and "n_eff in 7.7 %" in text


def test_relation_verdict_for_the_cointegrated_pair_lists_every_method():
    rows = [_rel("coint", 0.0, 104, naiv=1.0, diff=1.0, trend=1.0, neff=0.98, shift=1.0, dw=0.0)]
    kind, text = ev.verdict_relation(rows, "coint", 104)
    assert kind == "info" and "echter Niveaubeziehung" in text and "naiver Test 100.0 %" in text and "n_eff 98.0 %" in text and "Warnregel warnt in 0.0 %" in text


def test_variance_verdict_picks_the_row_where_the_formula_is_off_most():
    rows = [{"phi": 0.5, "T": 104, "t_var": 1.673, "formula": 1.667, "n_eff": 62.4}, {"phi": 0.99, "T": 104, "t_var": 21.2, "formula": 99.5, "n_eff": 1.0},
            {"phi": 0.9, "T": 520, "t_var": 8.9, "formula": 9.53, "n_eff": 54.6}]
    kind, text = ev.verdict_variance(rows)
    assert kind == "info" and "φ = 0.99 und T = 104" in text and "**21.20**" in text and "99.50" in text and "1/(1 − φ) = 100" in text and "1.0" in text
