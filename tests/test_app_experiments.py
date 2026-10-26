"""App: jedes Preset, der Permalink, jeder Experiment-Knopf einzeln und die Eckfall-Kombinationen (DEMO-PLAYBOOK §1d: jeder Knopf einzeln im AppTest)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import skz_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join([m.value for m in at.markdown] + [s.value for s in at.success] + [i.value for i in at.info] + [w.value for w in at.warning])


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_and_shows_its_data(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    preset = C.PRESETS[name]
    assert at.session_state["T_slider"] == preset["T"] and at.session_state["process_select"] == preset["process"] and at.session_state["rho_slider"] == preset["rho"]
    assert len(at.get("plotly_chart")) == 9                    # Hauptansicht 3, E1: 1, E2: 2, E3: 2, E4: 1; E5 erst nach dem Knopf


def test_permalink_is_loaded_and_clamped():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["proc"] = "ar1"
    at.query_params["T"] = "99999"
    at.query_params["phi"] = "5"
    at.query_params["rho"] = "abc"                # ungültig: wird ignoriert
    at.run()
    assert not at.exception
    assert at.session_state["process_select"] == "ar1" and at.session_state["T_slider"] == C.T_MAX and at.session_state["phi_slider"] == C.PHI_MAX and at.session_state["rho_slider"] == C.DEFAULT_RHO


def test_five_metrics_show_the_naive_result_and_the_warning_rule():
    at = _app()
    assert [m.label for m in at.metric] == ["Korrelation r", "p-Wert (naiv)", "R² der Regression", "Durbin–Watson", "Warnregel R² > DW"] and len(at.metric) == 5


def test_the_headline_verdict_changes_with_rho():
    at = _app()
    assert "unabhängig" in _texts(at) and "Ein Zusammenhang (ρ" not in _texts(at)
    at.sidebar.slider(key="rho_slider").set_value(0.9).run()
    assert not at.exception and "Ein Zusammenhang (ρ = 0.9) ist erzeugt" in _texts(at)


def test_pdf_button_then_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception and len(at.get("download_button")) == 1


def test_experiment_selectors_run():
    at = _app()
    for key, value in (("e1_method", "diff"), ("e1_method", "shift"), ("e1_T", 26), ("e1_T", 520), ("e2_process", "weiss"), ("e2_process", "trend"), ("e2_T", 26), ("e2_T", 520),
                       ("e3_T", 26), ("e3_T", 520), ("e3_process", "trend"), ("e3_process", "weiss"), ("e4_scenario", "ar1"), ("e4_scenario", "coint"), ("e4_scenario", "trend"), ("e4_T", 52), ("e4_T", 260)):
        at.selectbox(key=key).set_value(value).run()
        assert not at.exception, (key, value, [str(e) for e in at.exception])


def test_the_variance_button_shows_chart_table_and_verdict():
    at = _app()
    assert "T · Var(r) ist" not in _texts(at) and "T · Var(r) =" not in " ".join(i.value for i in at.info)
    at.button(key="btn_var").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    text = _texts(at)
    assert "| φ | T | T · Var(r) | Formel | effektive Stichprobengröße |" in text and "die Formel gilt erst, wenn T deutlich größer ist als die Gedächtnislänge" in text and "| 0.99 | 104 |" in text


def test_the_experiment_sections_show_their_verdicts():
    text = _texts(_app())
    for needle in ("Bei T = 104 Wochen schlägt der naive Test bei unabhängigen Irrfahrten", "Irrfahrt, T = 104: die Streuung von r ist", "gezogenen Paaren hat", "Bei T = 104 hält (Fehlalarm höchstens 7.5 %)",
                   "Differenzieren heilt die Irrfahrt", "Bei unabhängigen Irrfahrten warnt die Regel R² > DW", "zwei Irrfahrten, Zuwächse korreliert, T = 104"):
        assert needle in text, needle


def test_a_related_pair_changes_the_main_chart_text_and_keeps_the_page_running():
    at = _app()
    at.sidebar.button(key="preset_Echter Zusammenhang (Irrfahrten, ρ = 0,3)").click().run()
    assert not at.exception and "Ein Zusammenhang (ρ = 0.3) ist erzeugt" in _texts(at)


def test_corner_combination_of_all_minima_and_all_maxima():
    for edge in ("min", "max"):
        at = _app()
        for key in ("phi_slider", "slope_slider", "T_slider", "rho_slider"):
            w = at.sidebar.slider(key=key)
            value = w.min if edge == "min" else w.max
            w.set_value(int(value) if isinstance(w.value, int) else value)
        at.sidebar.selectbox(key="process_select").set_value("ar1" if edge == "min" else "trend")
        at.run()
        assert not at.exception, (edge, [str(e) for e in at.exception])
