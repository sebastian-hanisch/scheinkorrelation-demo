"""Auswertung: Einzelpaar, Zellen des vorgerechneten Rasters, kleine Live-Experimente, Zugriff auf das Raster und alle Verdikte als reine Funktionen."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import prs_algorithm as PA
import prs_scenario as S
import skz_algorithm as K
import skz_constants as C

GRID_PATH = Path(__file__).resolve().parent / "data" / "spurious_grid.json"
HOLD_LIMIT = 0.075            # „hält“: Fehlalarm bei unabhängigen Reihen höchstens 5 % plus 2,5 Punkte
RATE_KEYS = tuple(C.METHODS) + (C.WARNING_RULE,)


# --- Einzelpaar ------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    process: str = "walk"
    phi: float = C.DEFAULT_PHI
    slope: float = C.DEFAULT_SLOPE
    T: int = C.DEFAULT_T
    rho: float = C.DEFAULT_RHO
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    x: tuple[float, ...]
    y: tuple[float, ...]
    results: dict               # Verfahren -> (Statistik r, p-Wert)
    found: tuple[str, ...]      # Verfahren mit p ≤ 5 %
    dw: float
    r2: float
    warning: bool               # Granger–Newbold-Warnregel R² > DW
    related: bool               # ist ein Zusammenhang erzeugt (ρ > 0)?


def analyse(s: Settings) -> Analysis:
    X, Y = K.make_pair(s.process, 1, s.T, S.VRng(s.seed), s.rho, s.phi, s.slope)
    x, y = X[0], Y[0]
    out = K.test_pair(x, y)
    results = {m: out[m] for m in C.METHODS}
    found = tuple(m for m in C.METHODS if results[m][1] <= C.ALPHA)
    dw, r2, warn = out[C.WARNING_RULE]
    return Analysis(s, tuple(map(float, x)), tuple(map(float, y)), results, found, dw, r2, warn, s.rho > 0)


# --- Zellen des vorgerechneten Rasters -------------------------------------------------------------------------------------------------------

def cell_seed(kind: str, key: str, T: int, extra: float, s: int) -> int:
    """Eigener Seed je Zelle und Plan-Seed s (verschieden für Art, Prozess bzw. Szenario, T und den dritten Parameter: ρ oder φ)."""
    kinds = ("fa", "rel", "hist", "var", "probe")
    names = list(C.GRID_PROCESSES) + list(C.REL_SCENARIOS)
    return s * 1_000_003 + T * 101 + int(round(extra * 100)) * 17 + names.index(key) * 1013 + kinds.index(kind) * 100_000_007


def _rates(X, Y) -> dict:
    out = {m: float(K.reject_rows(m, X, Y).mean()) for m in C.METHODS}
    out[C.WARNING_RULE] = float(K.warning_rows(X, Y).mean())
    return out


def _aggregate(per_seed: list[dict]) -> tuple[dict, dict]:
    rate = {k: float(np.mean([p[k] for p in per_seed])) for k in per_seed[0]}
    sd = {k: float(np.std([p[k] for p in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0 for k in per_seed[0]}
    return rate, sd


def fa_cell(key: str, T: int, runs_per_seed: int = C.GRID_RUNS, seeds=C.GRID_SEEDS) -> dict:
    """E1 bis E3: Anteil verworfener Unabhängigkeit bei UNABHÄNGIGEN Reihen je Verfahren (und Anteil mit R² > DW), dazu die Verteilung von r (Mittel über die Seeds, Streuung über die Seeds)."""
    process, phi = C.GRID_PROCESSES[key]
    per_seed, rs = [], []
    for s in seeds:
        X, Y = K.make_pair(process, runs_per_seed, T, S.VRng(cell_seed("fa", key, T, 0.0, s)), 0.0, phi, C.TREND_SLOPE)
        per_seed.append(_rates(X, Y))
        rs.append(K.corr_rows(X, Y))
    rate, sd = _aggregate(per_seed)
    r = np.concatenate(rs)
    return {"process": key, "T": T, "runs": runs_per_seed * len(seeds), "rate": rate, "sd": sd,
            "r": {"sd": float(r.std()), "mean_abs": float(np.abs(r).mean()), "q05": float(np.quantile(r, 0.05)), "q95": float(np.quantile(r, 0.95)), "share_big": float((np.abs(r) > 0.5).mean())}}


def rel_cell(scenario: str, rho: float, T: int, runs_per_seed: int = C.GRID_RUNS, seeds=C.GRID_SEEDS) -> dict:
    """E4: Anteil verworfener Unabhängigkeit bei ECHTEM Zusammenhang (Innovationen mit Korrelation ρ; „coint“: y = x + Rauschen, ρ ohne Bedeutung)."""
    per_seed = []
    for s in seeds:
        rng = S.VRng(cell_seed("rel", scenario, T, rho, s))
        if scenario == "coint":
            X, Y = K.make_cointegrated_pair(runs_per_seed, T, rng, C.COINT_NOISE)
        else:
            X, Y = K.make_pair(scenario, runs_per_seed, T, rng, rho, C.REL_PHI, C.TREND_SLOPE)
        per_seed.append(_rates(X, Y))
    rate, sd = _aggregate(per_seed)
    return {"scenario": scenario, "rho": rho, "T": T, "runs": runs_per_seed * len(seeds), "rate": rate, "sd": sd}


def grid_probe() -> dict:
    """Kleine feste Zelle und ein fester Einzeldatensatz, die der Test nachrechnet: ändert sich Erzeuger oder Statistik, passen sie nicht mehr zum gespeicherten Raster."""
    a = analyse(Settings("walk", 0.9, 0.05, 60, 0.3, 11))
    return {"cell": fa_cell("walk", 40, 40, seeds=(7, 8)), "stats": {m: a.results[m][0] for m in C.METHODS}, "p": {m: a.results[m][1] for m in C.METHODS}, "dw": [a.dw, a.r2]}


# --- Live klein -----------------------------------------------------------------------------------------------------------------------------

def r_sample(key: str, T: int, runs: int = C.HIST_RUNS, seed: int = 7) -> np.ndarray:
    """E2: Pearson-r von `runs` unabhängigen Paaren des Prozesses (Schlüssel des Rasters)."""
    process, phi = C.GRID_PROCESSES[key]
    X, Y = K.make_pair(process, runs, T, S.VRng(cell_seed("hist", key, T, 0.0, seed)), 0.0, phi, C.TREND_SLOPE)
    return K.corr_rows(X, Y)


def r_critical(T: int, alpha: float = C.ALPHA) -> float:
    """Der Betrag von r, ab dem der naive t-Test bei T Beobachtungen verwirft: t/√(t² + T − 2)."""
    t = PA.t_critical(T - 2, alpha)
    return t / math.sqrt(t * t + T - 2)


def variance_formula(phi: float) -> float:
    """Grenzwert von T · Var(r) bei unabhängigen AR(1)-Reihen mit gleichem φ: (1 + φ²)/(1 − φ²)."""
    return (1.0 + phi * phi) / (1.0 - phi * phi)


def variance_cell(phi: float, T: int, runs_per_seed: int = C.VAR_RUNS, seeds=C.GRID_SEEDS) -> dict:
    """E5: T · Var(r) über unabhängige AR(1)-Paare gegen die Formel (und die daraus folgende effektive Stichprobengröße T/Formel)."""
    rs = []
    for s in seeds:
        X, Y = K.make_pair("ar1", runs_per_seed, T, S.VRng(cell_seed("var", "ar09", T, phi, s)), 0.0, phi)
        rs.append(K.corr_rows(X, Y))
    r = np.concatenate(rs)
    f = variance_formula(phi)
    return {"phi": phi, "T": T, "runs": len(r), "t_var": float(T * r.var()), "formula": f, "n_eff": T / f}


# --- Zugriff auf das Raster ------------------------------------------------------------------------------------------------------------------

def load_grid(path: Path = GRID_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def find(rows: list[dict], **keys) -> dict | None:
    for row in rows:
        if all(row.get(k) == v or (isinstance(v, float) and isinstance(row.get(k), (int, float)) and abs(row[k] - v) < 1e-9) for k, v in keys.items()):
            return row
    return None


def fa_table(grid: dict, T: int) -> dict:
    """Prozess -> Zelle für ein T."""
    return {key: find(grid["fa"], process=key, T=T) for key in C.GRID_PROCESSES}


def holds(cell: dict, method: str) -> bool:
    """Hält das Verfahren in dieser Zelle (unabhängige Reihen) sein Niveau?"""
    return cell["rate"][method] <= HOLD_LIMIT


# --- Verdikte (reine Funktionen: Art, Text) -----------------------------------------------------------------------------------------------------

def _names(keys) -> str:
    return ", ".join(C.METHODS[k] for k in keys)


def verdict_main(a: Analysis) -> tuple[str, str]:
    """Kernfrage der Hauptansicht: Welche Verfahren verwerfen die Unabhängigkeit, und stimmt das mit der Wahrheit überein?"""
    s = a.settings
    naive_r, naive_p = a.results["naiv"]
    base = f"Gemessen: r = **{naive_r:.2f}**, naiver p-Wert **{naive_p:.3f}** ({C.PROCESSES[s.process]}, T = {s.T} Wochen)."
    warn = f" Warnregel R² > DW: R² = {a.r2:.2f}, DW = {a.dw:.2f}, **{'Warnung' if a.warning else 'keine Warnung'}**."
    if not a.related:
        if not a.found:
            return "success", f"{base} Die Reihen sind unabhängig erzeugt, und kein Verfahren schlägt an.{warn}"
        if "naiv" in a.found and len(a.found) == 1:
            return "warning", f"{base} Die Reihen sind **unabhängig** erzeugt: der naive Test schlägt an, ein Fehlalarm, die Gegenmittel nicht.{warn}"
        return "warning", f"{base} Die Reihen sind **unabhängig** erzeugt, und es schlagen an: **{_names(a.found)}**, alles Fehlalarme.{warn}"
    if not a.found:
        return "info", f"{base} Ein Zusammenhang (ρ = {s.rho:g}) ist erzeugt, aber kein Verfahren findet ihn bei diesem T.{warn}"
    missed = [m for m in C.METHODS if m not in a.found]
    tail = f"; es übersehen ihn: {_names(missed)}" if missed else ""
    return "success", f"{base} Ein Zusammenhang (ρ = {s.rho:g}) ist erzeugt; es finden ihn: **{_names(a.found)}**{tail}.{warn}"


def verdict_fa(table: dict, T: int) -> tuple[str, str]:
    """E1: Fehlalarm des naiven Tests bei Irrfahrten und bei weißem Rauschen für ein T."""
    walk, white = table["walk"], table["weiss"]
    kind = "warning" if walk["rate"]["naiv"] > 0.2 else "info"
    return kind, (f"Bei T = {T} Wochen schlägt der naive Test bei unabhängigen Irrfahrten in **{100 * walk['rate']['naiv']:.1f} %** der {walk['runs']} Läufe an, bei weißem Rauschen in "
                  f"{100 * white['rate']['naiv']:.1f} % (erwartet 5 %).")


def verdict_r(cell: dict) -> tuple[str, str]:
    """E2: die Verteilung von r in einer Rasterzelle."""
    r = cell["r"]
    name = C.GRID_NAMES[cell["process"]]
    return "info", (f"{name}, T = {cell['T']}: die Streuung von r ist **{r['sd']:.3f}** (weißes Rauschen: 1/√T = {1 / math.sqrt(cell['T']):.3f}), das mittlere |r| {r['mean_abs']:.3f}, "
                    f"in **{100 * r['share_big']:.1f} %** der Paare ist |r| > 0,5.")


def verdict_hist(sample: np.ndarray, T: int) -> tuple[str, str]:
    """E2 live: Anteil der gezogenen Paare, die den naiven Test (|r| > r_krit) bestehen."""
    crit = r_critical(T)
    share = float((np.abs(sample) > crit).mean())
    return ("warning" if share > 0.1 else "success"), f"Von {len(sample)} gezogenen Paaren hat **{100 * share:.1f} %** ein |r| über der Schwelle {crit:.3f} des naiven Tests (bei Unabhängigkeit erwartet: 5 %)."


def verdict_methods(table: dict, T: int) -> tuple[str, str]:
    """E3: in wie vielen der sechs Prozesse jedes Verfahren sein Niveau hält (Fehlalarm ≤ 7,5 %)."""
    n = len(table)
    counts = {m: sum(holds(c, m) for c in table.values()) for m in C.METHODS}
    all_hold = [m for m in C.METHODS if counts[m] == n]
    text = (f"Bei T = {T} hält (Fehlalarm höchstens {100 * HOLD_LIMIT:.1f} %) " + ", ".join(f"{C.METHODS[m]} in **{counts[m]} von {n}**" for m in C.METHODS) + " Prozessen. "
            + (f"In allen: {_names(all_hold)}." if all_hold else "**Kein Verfahren hält in allen.**"))
    return ("success" if all_hold else "warning"), text


def verdict_overdiff(table: dict, T: int) -> tuple[str, str]:
    """E3: der Preis des Differenzierens bei weißem Rauschen und bei Trend plus Rauschen."""
    w, t = table["weiss"]["rate"]["diff"], table["trend"]["rate"]["diff"]
    return "info", f"Differenzieren heilt die Irrfahrt ({100 * table['walk']['rate']['diff']:.1f} %), hebt aber den Fehlalarm bei weißem Rauschen auf **{100 * w:.1f} %** und bei Trend plus Rauschen auf **{100 * t:.1f} %** (T = {T})."


def verdict_warning(grid: dict) -> tuple[str, str]:
    """E3: die Warnregel R² > DW bei unabhängigen Irrfahrten gegen T."""
    rows = sorted((find(grid["fa"], process="walk", T=T) for T in C.GRID_TS), key=lambda r: r["T"])
    first, last = rows[0], rows[-1]
    return "info", (f"Bei unabhängigen Irrfahrten warnt die Regel R² > DW in **{100 * first['rate']['dw']:.1f} %** der Läufe bei T = {first['T']} und in **{100 * last['rate']['dw']:.1f} %** bei T = {last['T']}; "
                    f"bei Trend plus Rauschen warnt sie in {100 * find(grid['fa'], process='trend', T=last['T'])['rate']['dw']:.1f} %.")


def verdict_relation(rows: list[dict], scenario: str, T: int) -> tuple[str, str]:
    """E4: bei ρ = 0 und ρ = 0,3: wer verwirft wie oft (naiver Test gegen das beste Gegenmittel)."""
    null, real = find(rows, scenario=scenario, rho=0.0, T=T), find(rows, scenario=scenario, rho=0.3, T=T)
    if scenario == "coint":
        r = null["rate"]
        return "info", f"Bei echter Niveaubeziehung (y = x + Rauschen, T = {T}) finden sie alle: " + ", ".join(f"{C.METHODS[m]} {100 * r[m]:.1f} %" for m in C.METHODS) + f"; die Warnregel warnt in {100 * r['dw']:.1f} %."
    best = max(C.METHODS, key=lambda m: real["rate"][m])
    return "info", (f"{C.REL_SCENARIOS[scenario]}, T = {T}: ohne Zusammenhang (ρ = 0) verwirft der naive Test in **{100 * null['rate']['naiv']:.1f} %**, mit ρ = 0,3 in **{100 * real['rate']['naiv']:.1f} %**; "
                    f"am besten findet ρ = 0,3 das Verfahren {C.METHODS[best]} (**{100 * real['rate'][best]:.1f} %**), n_eff in {100 * real['rate']['neff']:.1f} %.")


def verdict_variance(rows: list[dict]) -> tuple[str, str]:
    """E5: T·Var(r) gegen die Formel bei der kleinsten und der größten Stichprobe."""
    rows = sorted(rows, key=lambda r: (r["phi"], r["T"]))
    worst = max(rows, key=lambda r: r["formula"] / r["t_var"])
    return "info", (f"Bei φ = {worst['phi']:g} und T = {worst['T']} ist T · Var(r) = **{worst['t_var']:.2f}**, die Formel sagt {worst['formula']:.2f}: die Formel gilt erst, wenn T deutlich größer ist als die Gedächtnislänge 1/(1 − φ) = "
                    f"{1 / (1 - worst['phi']):.0f}; die effektive Stichprobengröße wäre dort {worst['n_eff']:.1f}.")
