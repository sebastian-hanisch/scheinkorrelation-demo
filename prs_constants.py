# Wörtliche Kopie von pearson-demo/prs_constants.py (letzter Commit der Datei: 35527c3, Arbeitsbaum sauber am 2026-10-25); keine Änderung außer dieser Zeile. Änderungen gehören ins Original, nicht hierher.
"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Komplexität am Regler-Maximum (DEMO-PLAYBOOK §1c): ein Datensatz mit n = 1 000 und B = 2 000 Bootstrap-Ziehungen kostet 2·10^6 Indizes
(O(B·n), rund 0,1 s); Wiederholungsexperimente folgen dem Budget REPS_BUDGET (Wiederholungen · B · n)."""

N_MIN, N_MAX, DEFAULT_N = 5, 1000, 30
RHO_MIN, RHO_MAX, DEFAULT_RHO = -0.95, 0.95, 0.5
LEV_COUNT_MAX, LEV_K_MAX = 2, 15.0                 # Hebelpunkte: Anzahl und Lage in Standardabweichungen
DEFAULT_LEV_COUNT, DEFAULT_LEV_K = 0, 5.0
B_MIN, B_MAX, DEFAULT_B = 200, 2000, 1000           # Bootstrap-Ziehungen für den einzelnen Datensatz
REPS_MIN, REPS_MAX, DEFAULT_REPS = 100, 2000, 500   # Wiederholungen im Abdeckungsexperiment
SEED_MAX, DEFAULT_SEED = 999999, 35
REPS_BUDGET = 10**8                                  # Wiederholungen · B · n höchstens (rund 4 s auf der Entwicklungsmaschine)
LEV_DIRS = {"gleichläufig": 1, "gegenläufig": -1}   # Richtung der Hebelpunkte: (k, k) oder (k, −k)
LEVEL = 0.95                                        # Konfidenzniveau aller Intervalle

# Familien: Schlüssel -> Anzeigename. t(ν) und "kontaminiert" haben einen gemeinsamen Skalenfaktor für beide Kennzahlen
# (Spitzenwochen heben beide zugleich); "tindep3" hat unabhängige Ränder und dient nur dem Fehlalarm-Vergleich (E5).
FAMILIES = {
    "normal": "Normal",
    "t10": "t(10), gemeinsame Skala",
    "t5": "t(5), gemeinsame Skala",
    "t3": "t(3), gemeinsame Skala",
    "kontaminiert": "kontaminiert (5 % Spitzenwochen, 5-fache Skala)",
    "lognormal05": "Lognormal σ = 0,5",
    "lognormal1": "Lognormal σ = 1",
}
INDEPENDENT_T3 = "tindep3"
ALL_FAMILIES = {**FAMILIES, INDEPENDENT_T3: "t(3), unabhängige Ränder"}
T_DF = {"t10": 10, "t5": 5, "t3": 3, "tindep3": 3}   # Freiheitsgrade (ganzzahlig: Chi-Quadrat als Summe von Quadraten)
LOGNORMAL_SIGMA = {"lognormal05": 0.5, "lognormal1": 1.0}
CONTAM_SHARE, CONTAM_SCALE = 0.05, 5.0

# Raster der vorgerechneten Messreihe (tools/make_grid.py) und der festen Pläne
GRID_NS = (10, 20, 30, 50, 100, 300, 1000)
GRID_RHOS = (0.0, 0.5, 0.9)
GRID_SEEDS = (101, 102, 103, 104)
GRID_BOOT = 400
SIZE_NS = (30, 100, 300)
SIZE_FAMILIES = ("normal", "tindep3", "t3", "t5", "t10", "kontaminiert", "lognormal1")
LEVERAGE_NS = (10, 30, 100, 300)
LEVERAGE_KS = (0, 2, 3, 4, 5, 6, 8, 10, 15)
LAW_NUS = (5, 6, 8, 10, 20)
LAW_N, LAW_RUNS_PER_SEED = 4000, 500                 # vorgerechnetes Varianzgesetz: n = 4 000, vier Seeds zu je 500 Läufen
UTILITY_RS = (0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99)

# Formen (E1)
SHAPE_N = 40
SHAPE_TARGET_MIN, SHAPE_TARGET_MAX, DEFAULT_SHAPE_TARGET = 0.1, 0.95, 0.8
SHAPE_SEEDS = tuple(range(1, 51))

COLOR_POINTS = "#4c78a8"
COLOR_LEVER = "#d95f02"
COLOR_LINE = "#2F6B65"
COLOR_TRUE = "#222222"
COLOR_FISHER = "#4c78a8"
COLOR_PERCENTILE = "#f58518"
COLOR_BCA = "#54a24b"

PRESETS = {
    "Standardfall (Voreinstellung)": {"family": "normal", "rho": 0.5, "n": 30, "lev_count": 0, "lev_k": 5.0, "seed": DEFAULT_SEED},
    "Schwerer Rand (t(3), n = 300)": {"family": "t3", "rho": 0.5, "n": 300, "lev_count": 0, "lev_k": 5.0, "seed": DEFAULT_SEED},
    "Spitzenwochen (kontaminiert, n = 100)": {"family": "kontaminiert", "rho": 0.5, "n": 100, "lev_count": 0, "lev_k": 5.0, "seed": DEFAULT_SEED},
    "Ein einzelner Punkt (k = 5, n = 30)": {"family": "normal", "rho": 0.0, "n": 30, "lev_count": 1, "lev_k": 5.0, "seed": DEFAULT_SEED},
    "Unkorreliert, aber nicht unabhängig (t(3), ρ = 0)": {"family": "t3", "rho": 0.0, "n": 100, "lev_count": 0, "lev_k": 5.0, "seed": DEFAULT_SEED},
}
# Die Hilfetexte nennen Zahlen der vorgerechneten Messreihe (data/pearson_grid.json) und der Experimente; tests/test_claims.py prüft jede davon.
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Normalverteilung, ρ = 0,5, n = 30: der Fall, in dem alle Verfahren ihr Versprechen halten sollten (Experiment 2: Fisher-z deckt in der Messreihe zu 95,3 %).",
    "Schwerer Rand (t(3), n = 300)": "t(3) mit gemeinsamer Skala, n = 300: Fisher-z deckt in der Messreihe nur zu 59,2 % statt 95 %, der Perzentil-Bootstrap zu 92,4 %.",
    "Spitzenwochen (kontaminiert, n = 100)": "5 % der Wochen mit 5-facher Streuung, n = 100: Fisher-z deckt in der Messreihe zu 57,5 %, der Perzentil-Bootstrap zu 91,2 %.",
    "Ein einzelner Punkt (k = 5, n = 30)": "Unabhängige Daten (ρ = 0) und ein Punkt bei fünf Standardabweichungen: Experiment 4 misst, dass der Test bei n = 30 in 84 % der Läufe anschlägt. Dieser Datensatz ist ein einzelner Lauf.",
    "Unkorreliert, aber nicht unabhängig (t(3), ρ = 0)": "t(3) mit gemeinsamer Skala, ρ = 0, n = 100: der Test auf ρ = 0 schlägt in der Messreihe in 32,6 % der Läufe an statt in 5 %.",
}
