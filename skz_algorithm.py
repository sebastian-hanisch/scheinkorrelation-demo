"""Kern: Zeitreihen-Prozesse, Paare mit einstellbarem Zusammenhang und die fünf Verfahren, die zwischen zwei Reihen „signifikant“ oder „nicht“ sagen.

Pearson, die Student-Verteilung und `rejects_rows` kommen aus prs_algorithm.py (wörtliche Kopie). Alles ist vektorisiert über Wiederholungen: Reihen haben die Form (Wiederholungen, T).
Zufall nur über einen übergebenen VRng (prs_scenario). Verfahren (METHODS): naiver t-Test (T − 2 Freiheitsgrade), Test auf die Differenzen (T − 1 Beobachtungen), Test nach Abzug der
linearen Trendgeraden (T Beobachtungen, wie der naive), t-Test mit effektiver Stichprobengröße n_eff = T (1 − r₁ₓ r₁ᵧ)/(1 + r₁ₓ r₁ᵧ) (nach oben durch T begrenzt, mindestens 3) und zirkuläres
Verschiebungstest über alle T Verschiebungen (p = Anteil der Verschiebungen mit |r| ≥ |r₀|, die Identität zählt mit)."""

import math

import numpy as np

import prs_algorithm as PA
import skz_constants as C

PROCESS_KEYS = tuple(C.PROCESSES)
METHOD_KEYS = tuple(C.METHODS)


# --- Prozesse ---------------------------------------------------------------------------------------------------------------------------------

def ar1_filter(e: np.ndarray, phi: float) -> np.ndarray:
    """AR(1) x_t = φ x_{t−1} + e_t mit stationärem Start x₀ = e₀/√(1 − φ²); Zeilen sind unabhängige Wiederholungen."""
    if not 0.0 <= phi < 1.0:
        raise ValueError("φ muss in [0, 1) liegen")
    x = np.empty_like(e)
    x[:, 0] = e[:, 0] / math.sqrt(1.0 - phi * phi)
    for t in range(1, e.shape[1]):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x


def process_from_innovations(process: str, e: np.ndarray, phi: float = C.DEFAULT_PHI, slope: float = C.DEFAULT_SLOPE) -> np.ndarray:
    """Reihe aus standardnormalen Innovationen: weißes Rauschen (e selbst), AR(1), Irrfahrt (Summe, x₀ = e₀) oder Trend slope·t plus Rauschen (t = 0, 1, …)."""
    if process == "weiss":
        return e.copy()
    if process == "ar1":
        return ar1_filter(e, phi)
    if process == "walk":
        return np.cumsum(e, axis=1)
    if process == "trend":
        return slope * np.arange(e.shape[1])[None, :] + e
    raise ValueError(f"unbekannter Prozess: {process}")


def make_pair(process: str, runs: int, T: int, rng, rho: float = 0.0, phi: float = C.DEFAULT_PHI, slope: float = C.DEFAULT_SLOPE) -> tuple[np.ndarray, np.ndarray]:
    """`runs` Paare (x, y) der Länge T. Die Innovationen beider Reihen haben die Korrelation ρ (0: unabhängig). Ziehungsreihenfolge: erst e₁, dann z; e₂ = ρ e₁ + √(1 − ρ²) z."""
    if T < 3:
        raise ValueError("mindestens drei Zeitpunkte nötig")
    if not 0.0 <= rho < 1.0:
        raise ValueError("ρ muss in [0, 1) liegen")
    e1 = rng.normal((runs, T))
    z = rng.normal((runs, T))
    e2 = rho * e1 + math.sqrt(1.0 - rho * rho) * z
    return process_from_innovations(process, e1, phi, slope), process_from_innovations(process, e2, phi, slope)


def make_cointegrated_pair(runs: int, T: int, rng, noise: float = C.COINT_NOISE) -> tuple[np.ndarray, np.ndarray]:
    """Gemeinsame Irrfahrt: x ist eine Irrfahrt, y = x + noise·ε (die Niveaus hängen zusammen). Ziehungsreihenfolge: erst die Innovationen von x, dann ε."""
    x = np.cumsum(rng.normal((runs, T)), axis=1)
    return x, x + noise * rng.normal((runs, T))


# --- Bausteine --------------------------------------------------------------------------------------------------------------------------------

def corr_rows(X, Y) -> np.ndarray:
    return PA.pearson_rows(X, Y)


def diff_rows(X) -> np.ndarray:
    """Erste Differenzen je Zeile (T − 1 Werte)."""
    return np.diff(np.asarray(X, dtype=float), axis=1)


def detrend_rows(X) -> np.ndarray:
    """Residuen der Regression auf Zeit (Achsenabschnitt und Steigung) je Zeile."""
    X = np.asarray(X, dtype=float)
    t = np.arange(X.shape[1], dtype=float)
    t -= t.mean()
    b = (X * t).sum(1, keepdims=True) / (t * t).sum()
    return X - X.mean(1, keepdims=True) - b * t


def lag1_rows(X) -> np.ndarray:
    """Autokorrelation zum Lag 1 je Zeile: Σ (x_t − x̄)(x_{t+1} − x̄) / Σ (x_t − x̄)² (Nenner über alle T Werte)."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Xc[:, 1:] * Xc[:, :-1]).sum(1) / (Xc * Xc).sum(1)


def n_eff_rows(X, Y) -> np.ndarray:
    """Effektive Stichprobengröße je Paar: T (1 − r₁ₓ r₁ᵧ)/(1 + r₁ₓ r₁ᵧ), höchstens T (negative Produkte machen die Reihen nicht „unabhängiger“), mindestens 3."""
    T = np.asarray(X).shape[1]
    prod = lag1_rows(X) * lag1_rows(Y)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(T * (1.0 - prod) / (1.0 + prod), 3.0, float(T))


def shift_p_rows(X, Y) -> np.ndarray:
    """Zirkuläres Verschiebungstest: alle T Verschiebungen von y gegen x (die Identität ist dabei); p = #{|r_s| ≥ |r₀|} / T. Über FFT in O(T log T) je Paar."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    T = X.shape[1]
    Xc, Yc = X - X.mean(1, keepdims=True), Y - Y.mean(1, keepdims=True)
    cc = np.fft.irfft(np.conj(np.fft.rfft(Xc, axis=1)) * np.fft.rfft(Yc, axis=1), n=T, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cc / np.sqrt((Xc * Xc).sum(1, keepdims=True) * (Yc * Yc).sum(1, keepdims=True))
        count = (np.abs(r) >= np.abs(r[:, :1]) - 1e-12).sum(1)
    return np.where(np.isfinite(r[:, 0]), count / T, np.nan)


def durbin_watson_rows(X, Y) -> tuple[np.ndarray, np.ndarray]:
    """(DW, R²) der Regression y ~ x je Paar: DW = Σ(u_t − u_{t−1})² / Σ u_t² mit den Residuen u; R² = r²."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Xc, Yc = X - X.mean(1, keepdims=True), Y - Y.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (Xc * Yc).sum(1, keepdims=True) / (Xc * Xc).sum(1, keepdims=True)
        u = Yc - b * Xc
        dw = (np.diff(u, axis=1) ** 2).sum(1) / (u * u).sum(1)
    r = corr_rows(X, Y)
    return dw, r * r


def warning_rows(X, Y) -> np.ndarray:
    """Granger–Newbold-Warnregel je Paar: R² > DW (ein Zeichen für Scheinregression, keine Testentscheidung)."""
    dw, r2 = durbin_watson_rows(X, Y)
    return r2 > dw


# --- Verfahren ---------------------------------------------------------------------------------------------------------------------------------

def _p_from_t_rows(t: np.ndarray, df: np.ndarray) -> np.ndarray:
    return np.array([PA.t_two_sided_p(float(a), float(d)) if math.isfinite(a) else (0.0 if a > 0 else float("nan")) for a, d in zip(np.abs(t), df)])


def method_stats(method: str, X, Y) -> tuple[np.ndarray, np.ndarray]:
    """(Statistik, p-Wert) des Verfahrens je Paar: die Statistik ist die Korrelation, auf der der Test beruht (bei n_eff und Verschiebung die der Rohreihen)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    T = X.shape[1]
    if method == "naiv":
        r = corr_rows(X, Y)
        return r, _p_from_t_rows(_t_stat(r, T), np.full(len(r), T - 2.0))
    if method == "diff":
        r = corr_rows(diff_rows(X), diff_rows(Y))
        return r, _p_from_t_rows(_t_stat(r, T - 1), np.full(len(r), T - 3.0))
    if method == "trend":
        r = corr_rows(detrend_rows(X), detrend_rows(Y))
        return r, _p_from_t_rows(_t_stat(r, T), np.full(len(r), T - 2.0))
    if method == "neff":
        r = corr_rows(X, Y)
        n_eff = n_eff_rows(X, Y)
        return r, _p_from_t_rows(_t_stat(r, n_eff), n_eff - 2.0)
    if method == "shift":
        return corr_rows(X, Y), shift_p_rows(X, Y)
    raise ValueError(f"unbekanntes Verfahren: {method}")


def _t_stat(r: np.ndarray, n) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return r * np.sqrt((np.asarray(n, dtype=float) - 2.0) / (1.0 - r * r))


def reject_rows(method: str, X, Y, alpha: float = C.ALPHA) -> np.ndarray:
    """Verwirft das Verfahren die Unabhängigkeit (p ≤ α), je Paar. Die ersten drei Verfahren nutzen den vektorisierten kritischen Wert, n_eff und Verschiebung den p-Wert."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    T = X.shape[1]
    if method == "naiv":
        return PA.rejects_rows(corr_rows(X, Y), T, alpha)
    if method == "diff":
        return PA.rejects_rows(corr_rows(diff_rows(X), diff_rows(Y)), T - 1, alpha)
    if method == "trend":
        return PA.rejects_rows(corr_rows(detrend_rows(X), detrend_rows(Y)), T, alpha)
    p = method_stats(method, X, Y)[1]
    with np.errstate(invalid="ignore"):
        return p <= alpha


def test_pair(x, y) -> dict:
    """Alle fünf Verfahren auf einem Paar: Methode -> (Statistik r, p-Wert); dazu die Warnregel unter dem Schlüssel „dw“ als (DW, R², Warnung)."""
    X = np.asarray(x, dtype=float)[None, :]
    Y = np.asarray(y, dtype=float)[None, :]
    if X.shape != Y.shape or X.shape[1] < 5:
        raise ValueError("x und y müssen gleich lang sein, mindestens fünf Zeitpunkte")
    out = {m: tuple(float(v[0]) for v in method_stats(m, X, Y)) for m in METHOD_KEYS}
    dw, r2 = durbin_watson_rows(X, Y)
    out[C.WARNING_RULE] = (float(dw[0]), float(r2[0]), bool(r2[0] > dw[0]))
    return out
