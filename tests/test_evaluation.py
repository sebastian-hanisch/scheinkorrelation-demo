"""Auswertung: Einzelpaar, Zellen, Seeds, Zugriff auf das Raster (die Verdikte prüft test_verdicts.py)."""

import numpy as np
import pytest

import prs_algorithm as PA
import prs_scenario as S
import skz_algorithm as K
import skz_constants as C
import skz_evaluation as ev


def test_analysis_is_deterministic_and_internally_consistent():
    s = ev.Settings("walk", 0.9, 0.05, 60, 0.3, 11)
    a = ev.analyse(s)
    assert a == ev.analyse(s) and a.settings == s and len(a.x) == len(a.y) == 60 and a.related
    X, Y = K.make_pair("walk", 1, 60, S.VRng(11), 0.3, 0.9, 0.05)
    assert a.x == tuple(map(float, X[0])) and a.y == tuple(map(float, Y[0]))
    out = K.test_pair(X[0], Y[0])
    assert all(a.results[m] == out[m] for m in C.METHODS) and a.dw == out["dw"][0] and a.r2 == out["dw"][1] and a.warning == out["dw"][2]
    assert a.found == tuple(m for m in C.METHODS if a.results[m][1] <= 0.05)


@pytest.mark.parametrize("field,value", [("process", "ar1"), ("slope", 0.1), ("T", 61), ("rho", 0.3), ("seed", 12)])
def test_every_setting_changes_the_analysis(field, value):
    base = ev.Settings("trend", 0.9, 0.05, 60, 0.0, 11)
    assert ev.analyse(base) != ev.analyse(ev.Settings(**{**base.__dict__, field: value}))


def test_phi_acts_only_on_ar1():
    ar = ev.Settings("ar1", 0.9, 0.05, 60, 0.0, 11)
    assert ev.analyse(ar) != ev.analyse(ev.Settings(**{**ar.__dict__, "phi": 0.5}))
    walk = ev.Settings("walk", 0.9, 0.05, 60, 0.0, 11)
    other = ev.analyse(ev.Settings(**{**walk.__dict__, "phi": 0.5}))
    assert ev.analyse(walk).x == other.x and ev.analyse(walk).y == other.y                       # bei der Irrfahrt ändert φ die Reihen nicht


def test_rho_zero_means_unrelated_and_positive_means_related():
    assert not ev.analyse(ev.Settings(rho=0.0)).related and ev.analyse(ev.Settings(rho=0.05)).related


def test_cell_seeds_are_distinct_across_kinds_keys_T_extra_and_seed():
    keys = list(dict.fromkeys(list(C.GRID_PROCESSES) + list(C.REL_SCENARIOS)))               # „walk“ und „trend“ heißen in beiden Listen gleich
    seeds = {ev.cell_seed(kind, key, T, extra, s) for kind in ("fa", "rel", "hist", "var", "probe") for key in keys for T in (26, 104, 520) for extra in (0.0, 0.3, 0.6) for s in C.GRID_SEEDS}
    assert len(seeds) == 5 * len(keys) * 3 * 3 * 4


def test_fa_cell_structure_reproducibility_and_bounds():
    a = ev.fa_cell("walk", 40, 30, seeds=(1, 2))
    assert a == ev.fa_cell("walk", 40, 30, seeds=(1, 2)) and a["runs"] == 60 and a["T"] == 40 and a["process"] == "walk"
    assert set(a["rate"]) == set(ev.RATE_KEYS) and set(a["sd"]) == set(ev.RATE_KEYS)
    assert all(0 <= v <= 1 and v * 60 == pytest.approx(round(v * 60)) for v in a["rate"].values())          # 2 Seeds mit je 30 Läufen
    assert set(a["r"]) == {"sd", "mean_abs", "q05", "q95", "share_big"} and a["r"]["q05"] < 0 < a["r"]["q95"] and 0 < a["r"]["mean_abs"] < 1
    assert a != ev.fa_cell("walk", 40, 30, seeds=(3, 4)) and a["rate"] != ev.fa_cell("weiss", 40, 30, seeds=(1, 2))["rate"]


def test_fa_cell_uses_the_process_and_phi_of_the_grid_key():
    ar09 = ev.fa_cell("ar09", 60, 100, seeds=(5,))
    ar05 = ev.fa_cell("ar05", 60, 100, seeds=(5,))
    assert ar09["rate"]["naiv"] > ar05["rate"]["naiv"] + 0.1 and ar09["r"]["sd"] > ar05["r"]["sd"]


def test_rel_cell_covers_every_scenario_and_orders_with_rho():
    low = ev.rel_cell("walk", 0.0, 60, 100, seeds=(1,))
    high = ev.rel_cell("walk", 0.6, 60, 100, seeds=(1,))
    assert high["rate"]["diff"] > low["rate"]["diff"] + 0.5 and high["scenario"] == "walk" and high["rho"] == 0.6 and high["runs"] == 100
    coint = ev.rel_cell("coint", 0.0, 60, 50, seeds=(1,))
    assert all(coint["rate"][m] >= 0.9 for m in ("naiv", "diff", "trend", "shift")) and coint["rate"]["dw"] == 0.0
    for sc in ("ar1", "trend"):
        assert set(ev.rel_cell(sc, 0.3, 40, 20, seeds=(1,))["rate"]) == set(ev.RATE_KEYS)


def test_r_sample_is_reproducible_and_matches_the_cell_process():
    a = ev.r_sample("walk", 60, 500)
    assert (a == ev.r_sample("walk", 60, 500)).all() and a.shape == (500,) and a.std() > 0.3
    assert ev.r_sample("weiss", 60, 500).std() < 0.2 and not (ev.r_sample("walk", 60, 500, seed=8) == a).all()


def test_r_critical_is_the_boundary_of_the_naive_test():
    for T in (26, 104, 520):
        c = ev.r_critical(T)
        assert PA.t_test_p(c, T) == pytest.approx(0.05, abs=1e-6) and 0 < c < 1
    assert ev.r_critical(26) > ev.r_critical(104) > ev.r_critical(520)


def test_variance_formula_and_cell():
    assert ev.variance_formula(0.0) == 1.0 and ev.variance_formula(0.5) == pytest.approx(5 / 3) and ev.variance_formula(0.9) == pytest.approx(1.81 / 0.19)
    c = ev.variance_cell(0.0, 200, 100, seeds=(1, 2))
    assert c == ev.variance_cell(0.0, 200, 100, seeds=(1, 2)) and c["runs"] == 200 and c["formula"] == 1.0 and c["n_eff"] == 200 and 0.7 < c["t_var"] < 1.4
    assert ev.variance_cell(0.9, 200, 100, seeds=(1, 2))["t_var"] > 3 * c["t_var"]


def test_find_matches_exact_and_float_keys_and_returns_none_otherwise():
    rows = [{"scenario": "a", "rho": 0.3, "T": 3}, {"scenario": "a", "rho": 0.6, "T": 3}]
    assert ev.find(rows, scenario="a", rho=0.6) is rows[1] and ev.find(rows, scenario="a", rho=0.3 + 1e-12) is rows[0]
    assert ev.find(rows, scenario="b") is None and ev.find(rows, scenario="a", rho=0.31) is None and ev.find(rows, T=3) is rows[0]


def test_load_grid_returns_none_for_a_missing_or_broken_file(tmp_path):
    assert ev.load_grid(tmp_path / "gibt-es-nicht.json") is None
    broken = tmp_path / "kaputt.json"
    broken.write_text("{nicht json", encoding="utf-8")
    assert ev.load_grid(broken) is None
    good = tmp_path / "gut.json"
    good.write_text('{"a": [1, 2]}', encoding="utf-8")
    assert ev.load_grid(good) == {"a": [1, 2]}


def test_fa_table_and_holds():
    grid = ev.load_grid()
    table = ev.fa_table(grid, 104)
    assert list(table) == list(C.GRID_PROCESSES) and all(c["T"] == 104 and c["process"] == k for k, c in table.items())
    assert ev.holds({"rate": {"naiv": 0.075}}, "naiv") and not ev.holds({"rate": {"naiv": 0.0751}}, "naiv")
