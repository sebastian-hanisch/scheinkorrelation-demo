"""Das vorgerechnete Raster (data/spurious_grid.json): Aufbau, Plausibilität und Übereinstimmung mit dem Erzeuger (Probe)."""

import pytest

import skz_constants as C
import skz_evaluation as ev


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None, "data/spurious_grid.json fehlt: python tools/make_grid.py"
    return g


def test_probe_still_matches_the_generator(grid):
    """Ändert sich Erzeuger, Statistik oder Seed-Ableitung, passt die gespeicherte Probe nicht mehr: dann tools/make_grid.py neu laufen lassen."""
    probe = ev.grid_probe()
    assert probe["cell"]["runs"] == grid["probe"]["cell"]["runs"] and probe["cell"]["rate"] == grid["probe"]["cell"]["rate"]            # Trefferanteile exakt
    assert probe["cell"]["sd"] == pytest.approx(grid["probe"]["cell"]["sd"], rel=1e-9) and probe["cell"]["r"] == pytest.approx(grid["probe"]["cell"]["r"], rel=1e-9)
    assert probe["stats"] == pytest.approx(grid["probe"]["stats"], rel=1e-9) and probe["p"] == pytest.approx(grid["probe"]["p"], rel=1e-9) and probe["dw"] == pytest.approx(grid["probe"]["dw"], rel=1e-9)
    assert 0 < sum(1 for v in grid["probe"]["cell"]["rate"].values() if 0 < v < 1)                                                      # die Probe sieht Treffer und Nichttreffer


def test_grid_has_every_cell_exactly_once(grid):
    def keys(rows, *names):
        return sorted(tuple(r[n] for n in names) for r in rows)

    assert keys(grid["fa"], "process", "T") == sorted((k, T) for k in C.GRID_PROCESSES for T in C.GRID_TS)
    expected = [(sc, rho, T) for T in C.REL_TS for sc in C.REL_SCENARIOS for rho in ((0.0,) if sc == "coint" else C.REL_RHOS)]
    assert keys(grid["rel"], "scenario", "rho", "T") == sorted(expected)


def test_grid_meta_and_run_counts(grid):
    assert grid["meta"]["seeds"] == list(C.GRID_SEEDS)
    assert all(r["runs"] == 4 * C.GRID_RUNS for r in grid["fa"] + grid["rel"])


def test_every_rate_is_a_probability_and_a_multiple_of_the_run_count(grid):
    for r in grid["fa"] + grid["rel"]:
        for k, v in r["rate"].items():
            assert 0.0 <= v <= 1.0 and 0.0 <= r["sd"][k] <= 0.5 and v * r["runs"] == pytest.approx(round(v * r["runs"]), abs=1e-6)
        assert set(r["rate"]) == set(ev.RATE_KEYS)


def test_white_noise_keeps_the_level_for_the_naive_test_and_the_unbiased_remedies(grid):
    for T in C.GRID_TS:
        r = ev.find(grid["fa"], process="weiss", T=T)["rate"]
        assert all(abs(r[m] - 0.05) < 0.012 for m in ("naiv", "trend", "neff", "shift")) and r["dw"] == 0.0, (T, r)


def test_differencing_inflates_white_noise_and_trend_but_heals_the_walk(grid):
    for T in C.GRID_TS:
        assert 0.09 < ev.find(grid["fa"], process="weiss", T=T)["rate"]["diff"] < 0.13
        assert 0.09 < ev.find(grid["fa"], process="trend", T=T)["rate"]["diff"] < 0.13 or T == 26
        assert abs(ev.find(grid["fa"], process="walk", T=T)["rate"]["diff"] - 0.05) < 0.012


def test_the_random_walk_false_alarm_grows_with_T_and_r_does_not_shrink(grid):
    rates = [ev.find(grid["fa"], process="walk", T=T)["rate"]["naiv"] for T in C.GRID_TS]
    assert rates == sorted(rates) and rates[0] > 0.45 and rates[-1] > 0.85
    sds = [ev.find(grid["fa"], process="walk", T=T)["r"]["sd"] for T in C.GRID_TS]
    assert max(sds) - min(sds) < 0.02 and 0.47 < min(sds)
    white = [ev.find(grid["fa"], process="weiss", T=T)["r"]["sd"] for T in C.GRID_TS]
    assert white == sorted(white, reverse=True)


def test_trend_plus_noise_flips_between_26_and_104_weeks(grid):
    rates = {T: ev.find(grid["fa"], process="trend", T=T)["rate"] for T in C.GRID_TS}
    assert rates[26]["naiv"] < 0.12 and 0.7 < rates[52]["naiv"] < 0.9 and all(rates[T]["naiv"] == 1.0 for T in (104, 260, 520))
    assert all(abs(rates[T]["trend"] - 0.05) < 0.012 for T in C.GRID_TS) and all(rates[T]["neff"] == 1.0 for T in (104, 260, 520))


def test_the_warning_rule_grows_with_T_for_walks_and_is_silent_for_trend_and_noise(grid):
    walk = [ev.find(grid["fa"], process="walk", T=T)["rate"]["dw"] for T in C.GRID_TS]
    assert walk == sorted(walk) and walk[0] < 0.15 and walk[-1] > 0.75
    assert all(ev.find(grid["fa"], process="trend", T=T)["rate"]["dw"] == 0.0 and ev.find(grid["fa"], process="weiss", T=T)["rate"]["dw"] == 0.0 for T in C.GRID_TS)


def test_the_shift_test_fails_for_long_memory_and_the_walk(grid):
    for key in ("ar099", "walk"):
        assert ev.find(grid["fa"], process=key, T=104)["rate"]["shift"] > 0.12
    assert ev.find(grid["fa"], process="ar09", T=104)["rate"]["shift"] < 0.075


def test_real_relations_are_found_by_differencing_and_missed_by_n_eff(grid):
    walk = ev.find(grid["rel"], scenario="walk", rho=0.3, T=104)["rate"]
    assert walk["diff"] > 0.85 and walk["neff"] < 0.12 and walk["naiv"] < walk["diff"]
    null = ev.find(grid["rel"], scenario="walk", rho=0.0, T=104)["rate"]
    assert null["naiv"] > 0.7 and null["diff"] < 0.075
    for T in C.REL_TS:
        coint = ev.find(grid["rel"], scenario="coint", rho=0.0, T=T)["rate"]
        assert all(coint[m] > 0.95 for m in C.METHODS) and coint["dw"] == 0.0


def test_verdicts_run_on_every_grid_row(grid):
    for T in C.GRID_TS:
        table = ev.fa_table(grid, T)
        assert ev.verdict_fa(table, T)[0] in ("info", "warning") and ev.verdict_methods(table, T)[0] in ("warning", "success") and ev.verdict_overdiff(table, T)[0] == "info"
        for key in C.GRID_PROCESSES:
            assert ev.verdict_r(ev.find(grid["fa"], process=key, T=T))[0] == "info"
    assert ev.verdict_warning(grid)[0] == "info"
    for T in C.REL_TS:
        for sc in C.REL_SCENARIOS:
            assert ev.verdict_relation(grid["rel"], sc, T)[0] == "info"
