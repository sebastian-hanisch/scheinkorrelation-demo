"""Abbildungen: jede wird gebaut, hat die erwarteten Spuren und Achsen."""

import math

import numpy as np
import pytest

import skz_constants as C
import skz_evaluation as ev
import skz_visualization as V
import prs_scenario as S
import skz_algorithm as K


@pytest.fixture(scope="module")
def grid():
    return ev.load_grid()


def test_every_chart_builds_and_locks_its_axes(grid):
    a = ev.analyse(ev.Settings())
    sample = ev.r_sample("walk", 60, 200)
    figs = [V.build_series(a.x, a.y), V.build_scatter(a.x, a.y), V.build_pvalues(a.results), V.build_fa_lines(grid, "naiv"), V.build_r_histogram(sample, 60, ev.r_critical(60)),
            V.build_r_spread(grid), V.build_method_heatmap(ev.fa_table(grid, 104), 104), V.build_method_lines(grid, "walk"), V.build_relation(grid["rel"], "walk", 104),
            V.build_relation(grid["rel"], "coint", 104), V.build_variance([ev.variance_cell(f, 100, 20, seeds=(1,)) for f in (0.0, 0.5)] + [ev.variance_cell(0.5, 200, 20, seeds=(1,))])]
    for fig in figs:
        assert fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange and len(fig.data) >= 1


def test_series_chart_has_one_line_per_series_with_weekly_x():
    fig = V.build_series([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    assert [t.name for t in fig.data] == ["x: Bestand Standort A", "y: Auftragseingang Standort B"] and list(fig.data[0].x) == [1, 2, 3] and list(fig.data[1].y) == [3.0, 2.0, 1.0]


def test_scatter_has_a_regression_line_unless_x_is_constant():
    x, y = np.arange(10.0), 2 * np.arange(10.0) + 1
    fig = V.build_scatter(x, y)
    assert [t.name for t in fig.data] == ["Wochen", "Regressionsgerade"] and list(fig.data[1].y) == pytest.approx([1.0, 19.0])
    assert len(V.build_scatter(np.ones(5), np.arange(5.0)).data) == 1


def test_pvalue_chart_colors_found_methods_and_floors_tiny_p_values():
    res = {"naiv": (0.5, 0.0), "diff": (0.1, 0.05), "trend": (0.2, 0.0501), "neff": (0.1, float("nan")), "shift": (0.0, 0.9)}
    fig = V.build_pvalues(res)
    assert list(fig.data[0].marker.color) == [C.COLOR_FOUND, C.COLOR_FOUND, C.COLOR_MISSED, C.COLOR_MISSED, C.COLOR_MISSED]      # p = 0,05 zählt als gefunden, NaN nicht
    assert list(fig.data[0].x) == [1e-4, 0.05, 0.0501, 1.0, 0.9] and list(fig.data[0].y) == list(C.METHODS.values())
    assert fig.data[0].text[0] == "< 0.001" and fig.data[0].text[1] == "0.050"


def test_fa_lines_have_one_line_per_process_sorted_by_T(grid):
    fig = V.build_fa_lines(grid, "diff")
    assert [t.name for t in fig.data][:6] == [C.GRID_NAMES[k] for k in C.GRID_PROCESSES] and list(fig.data[0].x) == list(C.GRID_TS)
    walk = next(t for t in fig.data if t.name == "Irrfahrt")
    assert list(walk.y) == pytest.approx([100 * ev.find(grid["fa"], process="walk", T=T)["rate"]["diff"] for T in C.GRID_TS])


def test_r_histogram_has_two_threshold_lines():
    fig = V.build_r_histogram(S.VRng(1).normal(100) / 3, 104, 0.19)
    assert len(fig.layout.shapes) == 2 and sorted(s.x0 for s in fig.layout.shapes) == [-0.19, 0.19]


def test_r_spread_has_the_reference_curve(grid):
    fig = V.build_r_spread(grid)
    assert [t.name for t in fig.data] == ["weißes Rauschen", "AR(1) 0,9", "Irrfahrt", "1/√T"] and list(fig.data[3].y) == pytest.approx([1 / math.sqrt(t) for t in C.GRID_TS])


def test_heatmap_shows_percent_with_the_right_orientation(grid):
    table = ev.fa_table(grid, 104)
    fig = V.build_method_heatmap(table, 104)
    z = np.array(fig.data[0].z)
    assert z.shape == (6, 5) and list(fig.data[0].y) == [C.GRID_NAMES[k] for k in C.GRID_PROCESSES] and list(fig.data[0].x) == list(C.METHODS.values())
    assert z[4, 1] == pytest.approx(100 * table["walk"]["rate"]["diff"]) and z[5, 2] == pytest.approx(100 * table["trend"]["rate"]["trend"]) and fig.layout.yaxis.autorange == "reversed"


def test_method_lines_include_the_warning_rule(grid):
    fig = V.build_method_lines(grid, "walk")
    assert [t.name for t in fig.data] == list(C.METHODS.values()) + ["Warnung R² > DW"] and list(fig.data[0].x) == list(C.GRID_TS)


def test_relation_chart_groups_by_rho_and_labels_the_cointegrated_case(grid):
    fig = V.build_relation(grid["rel"], "walk", 104)
    assert [t.name for t in fig.data] == list(C.METHODS.values()) and list(fig.data[0].x) == ["ρ = 0", "ρ = 0.3", "ρ = 0.6"]
    assert list(V.build_relation(grid["rel"], "coint", 104).data[0].x) == ["y = x + Rauschen"]


def test_variance_chart_has_the_formula_and_one_line_per_T():
    rows = [ev.variance_cell(f, T, 20, seeds=(1,)) for T in (100, 200) for f in (0.0, 0.5)]
    fig = V.build_variance(rows)
    assert fig.data[0].name.startswith("Formel") and [t.name for t in fig.data[1:]] == ["Simulation, T = 100", "Simulation, T = 200"] and fig.layout.yaxis.type == "log"
    assert K.PROCESS_KEYS == ("weiss", "ar1", "walk", "trend")
