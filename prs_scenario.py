# Wörtliche Kopie von pearson-demo/prs_scenario.py (letzter Commit der Datei: 35527c3, Arbeitsbaum sauber am 2026-10-25); keine Änderung außer dieser Zeile. Änderungen gehören ins Original, nicht hierher.
"""Szenario: Paare (x, y) einer Standort-Woche (x: Abweichung des Auftragsvolumens vom Plan, y: Überstunden) mit bekannter wahrer Korrelation.

Zufall kommt nur aus SplitMix64 (prs_rng.py, wörtliche Kopie). SplitMix64 ist zählerbasiert: der k-te Wert ist mix(Seed + k·γ). Der vektorisierte Generator
VRng rechnet diese Werte für viele k zugleich und liefert genau den Strom der skalaren Klasse (tests/test_scenario.py prüft Gleichheit); so bleibt jede
Messreihe auf jeder Plattform gleich, ohne Python-Schleife über einzelne Zufallszahlen.

Familien (prs_constants.FAMILIES): Normal; t(ν) und kontaminiert mit GEMEINSAMEM Skalenfaktor für beide Kennzahlen (eine Spitzenwoche hebt beide zugleich,
die Korrelation bleibt ρ, die Kennzahlen sind aber nicht unabhängig); Lognormal (Mengen, schief; die wahre Korrelation weicht von ρ ab)."""

import math
from dataclasses import dataclass

import numpy as np

import prs_constants as C
from prs_rng import MASK

GAMMA = 0x9E3779B97F4A7C15
_U = np.uint64


class VRng:
    """Vektorisierter SplitMix64: u64(count) liefert die nächsten `count` Werte des Stroms, in derselben Reihenfolge wie SplitMix64(seed).next()."""

    def __init__(self, seed: int):
        self.seed = seed & MASK
        self.pos = 0

    def u64(self, count: int) -> np.ndarray:
        k = np.arange(self.pos + 1, self.pos + 1 + count, dtype=np.uint64)
        z = _U(self.seed) + k * _U(GAMMA)
        z = (z ^ (z >> _U(30))) * _U(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> _U(27))) * _U(0x94D049BB133111EB)
        z = z ^ (z >> _U(31))
        self.pos += count
        return z

    def uniform(self, shape) -> np.ndarray:
        """Gleichverteilt in [0, 1) mit 53 Bit."""
        size = int(np.prod(shape))
        return ((self.u64(size) >> _U(11)).astype(np.float64) * 2.0**-53).reshape(shape)

    def normal(self, shape) -> np.ndarray:
        """Standardnormal nach Box-Muller: erst m Werte u1 (in (0, 1]), dann m Werte u2; die Ausgabe ist erst alle Kosinus-, dann alle Sinus-Werte."""
        size = int(np.prod(shape))
        m = (size + 1) // 2
        u1 = 1.0 - self.uniform(m)
        u2 = self.uniform(m)
        rad = np.sqrt(-2.0 * np.log(u1))
        ang = 2.0 * math.pi * u2
        return np.concatenate([rad * np.cos(ang), rad * np.sin(ang)])[:size].reshape(shape)

    def below(self, k: int, shape) -> np.ndarray:
        """Gleichverteilte ganze Zahlen in [0, k)."""
        return np.minimum((self.uniform(shape) * k).astype(np.int64), k - 1)


def substream_seeds(seed: int, count: int) -> list[int]:
    """Unabhängige Seeds für `count` Wiederholungen: die gemischten Ausgaben eines Stroms (nie Seed + i, denn Ströme mit um γ verschobenem Seed überlappen)."""
    return [int(v) for v in VRng(seed).u64(count)]


def true_r(family: str, rho: float) -> float:
    """Wahre Pearson-Korrelation der Familie bei Parameter ρ."""
    if family in C.LOGNORMAL_SIGMA:
        s2 = C.LOGNORMAL_SIGMA[family] ** 2
        return (math.exp(s2 * rho) - 1.0) / (math.exp(s2) - 1.0)
    if family == C.INDEPENDENT_T3:
        return 0.0
    return rho


def t_df(family: str) -> int | None:
    """Freiheitsgrade, wenn die Familie t(ν) heißt: "t3", "t5", "t10", beliebiges "t<ν>" mit ν ≥ 3 (für das Varianzgesetz) und "tindep3"; sonst None."""
    if family in C.T_DF:
        return C.T_DF[family]
    if family.startswith("t") and family[1:].isdigit() and int(family[1:]) >= 3:
        return int(family[1:])
    return None


def draw_pairs(family: str, reps: int, n: int, rho: float, rng: VRng) -> tuple[np.ndarray, np.ndarray]:
    """`reps` Stichproben der Größe n als Arrays der Form (reps, n). Zuerst die beiden Normalfelder, danach (falls nötig) die Skalenfaktoren."""
    if family not in C.ALL_FAMILIES and t_df(family) is None:
        raise ValueError(f"unbekannte Familie: {family}")
    if not -1.0 < rho < 1.0:
        raise ValueError("rho muss zwischen -1 und 1 liegen")
    shape = (reps, n)
    z1 = rng.normal(shape)
    z2 = rho * z1 + math.sqrt(1.0 - rho * rho) * rng.normal(shape)
    if family == "normal":
        return z1, z2
    if family in C.LOGNORMAL_SIGMA:
        s = C.LOGNORMAL_SIGMA[family]
        return np.exp(s * z1), np.exp(s * z2)
    if family == "kontaminiert":
        w = np.where(rng.uniform(shape) < C.CONTAM_SHARE, C.CONTAM_SCALE, 1.0)
        return z1 * w, z2 * w
    nu = t_df(family)
    if family == C.INDEPENDENT_T3:  # unabhängige Ränder: jede Kennzahl mit eigenem Skalenfaktor (rho wird zu 0 erzwungen)
        z2 = rng.normal(shape)
        w1 = np.sqrt(nu / (rng.normal((nu,) + shape) ** 2).sum(0))
        w2 = np.sqrt(nu / (rng.normal((nu,) + shape) ** 2).sum(0))
        return z1 * w1, z2 * w2
    w = np.sqrt(nu / (rng.normal((nu,) + shape) ** 2).sum(0))
    return z1 * w, z2 * w


def add_leverage(x: np.ndarray, y: np.ndarray, count: int, k: float, direction: int) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Setzt die ersten `count` Beobachtungen auf (k, direction·k) in Einheiten der Standardabweichung der übrigen Daten; gibt auch die Indizes zurück.
    Hebelpunkte ersetzen Beobachtungen (n bleibt), es bleiben mindestens drei unveränderte Beobachtungen."""
    count = int(min(count, max(len(x) - 3, 0)))
    if count <= 0 or k == 0:
        return x.copy(), y.copy(), ()
    x2, y2 = x.copy(), y.copy()
    rest_x, rest_y = x[count:], y[count:]
    sx, sy = float(rest_x.std()), float(rest_y.std())
    mx, my = float(rest_x.mean()), float(rest_y.mean())
    for i in range(count):
        x2[i] = mx + k * sx
        y2[i] = my + direction * k * sy
    return x2, y2, tuple(range(count))


@dataclass(frozen=True)
class Dataset:
    x: tuple[float, ...]
    y: tuple[float, ...]
    lever: tuple[int, ...]    # Indizes der Hebelpunkte
    true_r: float


def make_dataset(family: str, n: int, rho: float, seed: int, lev_count: int = 0, lev_k: float = 0.0, lev_dir: int = 1) -> Dataset:
    """Ein Datensatz, deterministisch aus den Eingaben (gleiche Eingabe, gleicher Datensatz)."""
    x, y = draw_pairs(family, 1, n, rho, VRng(seed))
    x, y, lever = add_leverage(x[0], y[0], lev_count, lev_k, lev_dir)
    return Dataset(tuple(map(float, x)), tuple(map(float, y)), lever, true_r(family, rho))
