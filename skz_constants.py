"""Konstanten: Regler-Grenzen, Prozesse und Verfahren, Raster der Messreihen, Farben und Presets.

Zufallserzeuger und Pearson-Kern kommen als wörtliche Kopien aus pearson-demo (prs_rng.py, prs_constants.py, prs_scenario.py, prs_algorithm.py).
Rechenbudget am Regler-Maximum (DEMO-PLAYBOOK §1c): ein Paar mit T = 1 000 braucht für alle Verfahren Millisekunden (Verschiebungen per FFT in O(T log T));
die Wiederholungsexperimente stehen vorgerechnet im Raster (data/spurious_grid.json), live laufen nur die kleinen (E2: 2 000 Paare, E5: 4 · 250 Läufe je φ)."""

from prs_constants import SEED_MAX  # noqa: F401  (Wiederverwendung der Kopie)

T_MIN, T_MAX, DEFAULT_T = 20, 1000, 104
PHI_MIN, PHI_MAX, DEFAULT_PHI = 0.0, 0.99, 0.9
SLOPE_MIN, SLOPE_MAX, DEFAULT_SLOPE = 0.0, 0.2, 0.05
RHO_MIN, RHO_MAX, DEFAULT_RHO = 0.0, 0.9, 0.0
DEFAULT_SEED = 35
ALPHA = 0.05

# Prozesse der Reglerseite: Schlüssel -> Anzeigename
PROCESSES = {
    "weiss": "weißes Rauschen",
    "ar1": "AR(1) (mit φ)",
    "walk": "Irrfahrt",
    "trend": "Trend plus Rauschen",
}
# Verfahren: Schlüssel -> Anzeigename
METHODS = {
    "naiv": "naiver Test",
    "diff": "Differenzieren",
    "trend": "Trend weg",
    "neff": "n_eff",
    "shift": "Verschiebung",
}
METHOD_LONG = {
    "naiv": "naiver t-Test der Pearson-Korrelation",
    "diff": "Test auf die Korrelation der Differenzen",
    "trend": "Test nach Abzug der Trendgeraden",
    "neff": "t-Test mit effektiver Stichprobengröße",
    "shift": "zirkuläres Verschiebungstest",
}
WARNING_RULE = "dw"        # Granger–Newbold-Regel: R² > Durbin–Watson (keine Testentscheidung, ein Warnsignal)

# Raster der Messreihen (tools/make_grid.py): Prozess-Schlüssel -> (Prozess, φ)
GRID_PROCESSES = {
    "weiss": ("weiss", 0.0),
    "ar05": ("ar1", 0.5),
    "ar09": ("ar1", 0.9),
    "ar099": ("ar1", 0.99),
    "walk": ("walk", 0.0),
    "trend": ("trend", 0.0),
}
GRID_NAMES = {"weiss": "weißes Rauschen", "ar05": "AR(1) 0,5", "ar09": "AR(1) 0,9", "ar099": "AR(1) 0,99", "walk": "Irrfahrt", "trend": "Trend + Rauschen"}
GRID_SEEDS = (101, 102, 103, 104)
GRID_TS = (26, 52, 104, 260, 520)
GRID_RUNS = 1000                                    # Läufe je Seed
REL_SCENARIOS = {"walk": "zwei Irrfahrten, Zuwächse korreliert", "ar1": "zwei AR(1) 0,9, Innovationen korreliert", "trend": "gemeinsamer Trend, Rauschen korreliert",
                 "coint": "gemeinsame Irrfahrt, y = x + Rauschen"}
REL_RHOS = (0.0, 0.3, 0.6)
REL_TS = (52, 104, 260)
REL_PHI = 0.9
COINT_NOISE = 1.0
TREND_SLOPE = 0.05

# Live klein: Verteilung von r (E2) und Varianz von r (E5)
HIST_RUNS = 2000
VAR_PHIS = (0.0, 0.5, 0.9, 0.99)
VAR_TS = (104, 520)
VAR_RUNS = 250                                      # Läufe je Seed

COLOR_X = "#4c78a8"
COLOR_Y = "#f58518"
COLOR_TRUE = "#222222"
COLOR_FOUND = "#d95f02"
COLOR_MISSED = "#b0b0b0"
PROCESS_COLORS = {"weiss": "#9d9d9d", "ar05": "#9ecae1", "ar09": "#4c78a8", "ar099": "#08519c", "walk": "#e45756", "trend": "#54a24b"}
METHOD_COLORS = {"naiv": "#e45756", "diff": "#54a24b", "trend": "#b279a2", "neff": "#4c78a8", "shift": "#f58518"}

PRESETS = {
    "Standardfall (Voreinstellung)": {"process": "walk", "phi": DEFAULT_PHI, "slope": DEFAULT_SLOPE, "T": 104, "rho": 0.0, "seed": DEFAULT_SEED},
    "Weißes Rauschen (Kontrolle)": {"process": "weiss", "phi": DEFAULT_PHI, "slope": DEFAULT_SLOPE, "T": 104, "rho": 0.0, "seed": DEFAULT_SEED},
    "AR(1) mit φ = 0,9": {"process": "ar1", "phi": 0.9, "slope": DEFAULT_SLOPE, "T": 104, "rho": 0.0, "seed": DEFAULT_SEED},
    "Trend und Rauschen": {"process": "trend", "phi": DEFAULT_PHI, "slope": DEFAULT_SLOPE, "T": 104, "rho": 0.0, "seed": DEFAULT_SEED},
    "Echter Zusammenhang (Irrfahrten, ρ = 0,3)": {"process": "walk", "phi": DEFAULT_PHI, "slope": DEFAULT_SLOPE, "T": 104, "rho": 0.3, "seed": DEFAULT_SEED},
    "Überdifferenziert (weißes Rauschen, 260 Wochen)": {"process": "weiss", "phi": DEFAULT_PHI, "slope": DEFAULT_SLOPE, "T": 260, "rho": 0.0, "seed": DEFAULT_SEED},
}
# Die Hilfetexte nennen Zahlen des vorgerechneten Rasters (data/spurious_grid.json); tests/test_claims.py prüft jede davon.
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Zwei unabhängige Irrfahrten, 104 Wochen: der naive Test schlägt in der Messreihe in 76,1 % der Läufe an, Differenzieren in 4,4 % und die n_eff-Korrektur in 4,6 %.",
    "Weißes Rauschen (Kontrolle)": "Weißes Rauschen, 104 Wochen: der naive Test hält seine 5 % (4,8 % der Läufe), Differenzieren hebt den Fehlalarm auf 11,3 %: Überdifferenzieren ist kein Gratis-Gegenmittel.",
    "AR(1) mit φ = 0,9": "AR(1) mit φ = 0,9, 104 Wochen: der naive Test schlägt in 50,4 % der Läufe an, Differenzieren in 5,1 %, n_eff in 4,7 %, das Verschiebungstest in 5,5 %.",
    "Trend und Rauschen": "Trend plus Rauschen, 104 Wochen: der naive Test schlägt in 100,0 % der Läufe an, n_eff ebenfalls in 100,0 %, nach Abzug der Trendgeraden sind es 5,0 %.",
    "Echter Zusammenhang (Irrfahrten, ρ = 0,3)": "Zwei Irrfahrten mit korrelierten Zuwächsen (ρ = 0,3), 104 Wochen: der naive Test schlägt in 79,4 % der Läufe an (ohne Zusammenhang 76,1 %), Differenzieren in 88,2 %, n_eff nur in 8,6 %.",
    "Überdifferenziert (weißes Rauschen, 260 Wochen)": "Weißes Rauschen, 260 Wochen: der naive Test schlägt in 4,9 % der Läufe an, nach dem Differenzieren in 11,8 %.",
}
