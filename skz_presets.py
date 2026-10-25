"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio; wie in pearson-demo, rangkorrelation-demo und abhaengigkeitsmasse-demo)."""

import random
from dataclasses import dataclass

import streamlit as st

import skz_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "process_select": SettingSpec("proc", str, "walk", choices=tuple(C.PROCESSES)),
    "phi_slider": SettingSpec("phi", float, C.DEFAULT_PHI, C.PHI_MIN, C.PHI_MAX),
    "slope_slider": SettingSpec("slope", float, C.DEFAULT_SLOPE, C.SLOPE_MIN, C.SLOPE_MAX),
    "T_slider": SettingSpec("T", int, C.DEFAULT_T, C.T_MIN, C.T_MAX),
    "rho_slider": SettingSpec("rho", float, C.DEFAULT_RHO, C.RHO_MIN, C.RHO_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"process": "process_select", "phi": "phi_slider", "slope": "slope_slider", "T": "T_slider", "rho": "rho_slider", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def clean_value(spec: SettingSpec, raw):
    """Wert aus der Adresszeile: umwandeln, auf Reglergrenzen begrenzen, Gleitkommazahlen auf zwei Stellen runden; ungültig gibt None."""
    try:
        value = spec.caster(raw)
    except (ValueError, TypeError):
        return None
    if spec.choices is not None:
        return value if value in spec.choices else None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return round(value, 2) if isinstance(value, float) else value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            value = clean_value(spec, qp[spec.url_param])
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Setzt ALLE Regler: die des Presets auf seine Werte, die übrigen auf ihren Standard (so bleibt die Hervorhebung des Presets nach dem Anklicken gültig)."""
    preset = C.PRESETS[name]
    key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = preset.get(key_of.get(state_key), spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
