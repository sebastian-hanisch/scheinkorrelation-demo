"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Hilfetexte, Adresszeile."""

import pytest

import skz_constants as C
import skz_evaluation as ev
import skz_presets as P


def test_presets_are_complete():
    assert "Standardfall (Voreinstellung)" in C.PRESETS
    assert set(C.PRESET_HELP) == set(C.PRESETS)
    for name, preset in C.PRESETS.items():
        assert C.PRESET_HELP[name] and C.PRESET_HELP[name] != name         # ein echter Text, nicht der Name
        assert set(preset) == set(P.PRESET_KEYS), name
        assert preset["process"] in C.PROCESSES
        assert C.PHI_MIN <= preset["phi"] <= C.PHI_MAX and C.SLOPE_MIN <= preset["slope"] <= C.SLOPE_MAX and C.T_MIN <= preset["T"] <= C.T_MAX
        assert C.RHO_MIN <= preset["rho"] <= C.RHO_MAX and 0 <= preset["seed"] <= C.SEED_MAX


def test_the_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_the_defaults_match():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    s = ev.Settings()
    assert [P.SETTING_SPECS[k].default for k in ("process_select", "phi_slider", "slope_slider", "T_slider", "rho_slider", "seed_input")] == [s.process, s.phi, s.slope, s.T, s.rho, s.seed]


def test_every_preset_differs_from_every_other():
    assert len({tuple(sorted(p.items())) for p in C.PRESETS.values()}) == len(C.PRESETS)


@pytest.mark.parametrize("raw,expected", [("99999", C.T_MAX), ("1", C.T_MIN), ("150", 150), ("abc", None), ("", None)])
def test_clean_value_clamps_integers(raw, expected):
    assert P.clean_value(P.SETTING_SPECS["T_slider"], raw) == expected


@pytest.mark.parametrize("raw,expected", [("0.123", 0.12), ("5", C.PHI_MAX), ("-1", 0.0), ("x", None)])
def test_clean_value_clamps_and_rounds_floats(raw, expected):
    assert P.clean_value(P.SETTING_SPECS["phi_slider"], raw) == expected


def test_clean_value_accepts_only_known_choices():
    spec = P.SETTING_SPECS["process_select"]
    assert P.clean_value(spec, "ar1") == "ar1" and P.clean_value(spec, "kreis") is None and P.clean_value(spec, "") is None
    assert P.clean_value(P.SETTING_SPECS["seed_input"], "-5") == 0 and P.clean_value(P.SETTING_SPECS["seed_input"], str(C.SEED_MAX + 1)) == C.SEED_MAX
