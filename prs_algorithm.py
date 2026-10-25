# Wörtliche Kopie von pearson-demo/prs_algorithm.py (letzter Commit der Datei: 35527c3, Arbeitsbaum sauber am 2026-10-25); keine Änderung außer dieser Zeile. Änderungen gehören ins Original, nicht hierher.
"""Kern: Pearson-Korrelation, Intervalle (Fisher, Perzentil-Bootstrap, BCa), t-Test, Permutationstest und die geschlossenen Formeln.

Alles in numpy und math, ohne scipy zur Laufzeit: Normal- und Student-Verteilung stehen hier (Fehlerfunktion, regularisierte unvollständige Beta-Funktion
als Kettenbruch). Jede Einheit ist eine eigene Funktion mit expliziten Ein- und Ausgaben; Zufall nur über einen übergebenen VRng (prs_scenario)."""

import math

import numpy as np

import prs_constants as C


# --- Verteilungen ---------------------------------------------------------------------------------------------------------------------------

def norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def norm_ppf(p: float) -> float:
    """Quantilfunktion der Standardnormalverteilung: Startwert nach Acklam, danach Newton-Schritte auf der Fehlerfunktion (Genauigkeit ~1e-15)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p muss in (0, 1) liegen")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p > 1.0 - 0.02425:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    for _ in range(3):
        err = norm_cdf(x) - p
        x -= err * math.sqrt(2.0 * math.pi) * math.exp(0.5 * x * x)
    return x


def _betacf(a: float, b: float, x: float) -> float:
    """Kettenbruch der unvollständigen Beta-Funktion (modifiziertes Lentz-Verfahren)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc_reg(a: float, b: float, x: float) -> float:
    """Regularisierte unvollständige Beta-Funktion I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """Zweiseitiger p-Wert der Student-Verteilung."""
    return betainc_reg(df / 2.0, 0.5, df / (df + t * t))


# --- Korrelation ----------------------------------------------------------------------------------------------------------------------------

def pearson_r(x, y) -> float:
    """Pearson-Korrelation (zwei Durchgänge: erst zentrieren, dann Kreuzprodukt durch Wurzel aus dem Produkt der Quadratsummen)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x und y müssen gleich lang sein")
    if len(x) < 2:
        raise ValueError("mindestens zwei Beobachtungen nötig")
    xc, yc = x - x.mean(), y - y.mean()
    sxx, syy = float((xc * xc).sum()), float((yc * yc).sum())
    if sxx == 0.0 or syy == 0.0:
        raise ValueError("eine Kennzahl ist konstant: die Korrelation ist nicht definiert")
    return max(-1.0, min(1.0, float((xc * yc).sum()) / math.sqrt(sxx * syy)))


def pearson_rows(X, Y) -> np.ndarray:
    """Zeilenweise Korrelation zweier Felder gleicher Form (reps, n): die vektorisierte Fassung von pearson_r für Wiederholungen. Konstante Zeilen geben NaN."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Xc = X - X.mean(1, keepdims=True)
    Yc = Y - Y.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip((Xc * Yc).sum(1) / np.sqrt((Xc * Xc).sum(1) * (Yc * Yc).sum(1)), -1.0, 1.0)


def slope(x, y) -> float:
    """Steigung der Regressionsgeraden y ~ x (gleich r · s_y / s_x)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


def rank_average(v) -> np.ndarray:
    """Ränge 1..n, Gleichstände mit dem Mittel ihrer Plätze."""
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v))
    sorted_v = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman_r(x, y) -> float:
    """Spearman-Korrelation als Pearson-Korrelation der Ränge (nur als Spalte zum Vergleich; erklärt wird sie in Stück 2)."""
    return pearson_r(rank_average(x), rank_average(y))


def jackknife_r(x, y) -> np.ndarray:
    """r ohne Beobachtung i, für alle i, in O(n) aus den Summen."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    m = n - 1
    ax, ay = x.sum() - x, y.sum() - y
    cxy = ((x * y).sum() - x * y) - ax * ay / m
    cxx = ((x * x).sum() - x * x) - ax * ax / m
    cyy = ((y * y).sum() - y * y) - ay * ay / m
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(cxy / np.sqrt(cxx * cyy), -1.0, 1.0)


def influence(x, y) -> np.ndarray:
    """Änderung von r, wenn Beobachtung i fehlt: r − r_(−i). Das Vorzeichen zeigt, ob der Punkt r hebt (positiv) oder senkt (negativ)."""
    return pearson_r(x, y) - jackknife_r(x, y)


# --- Tests und Intervalle ---------------------------------------------------------------------------------------------------------------------

def t_statistic(r: float, n: int) -> float:
    if n < 3:
        raise ValueError("der t-Test braucht mindestens drei Beobachtungen")
    return math.inf if abs(r) >= 1.0 else r * math.sqrt((n - 2) / (1.0 - r * r))


def t_critical(df: float, alpha: float = 0.05) -> float:
    """Kritischer Betrag der Student-Verteilung (zweiseitig) durch Bisektion auf dem p-Wert; Grundlage für vektorisierte Tests vieler Wiederholungen."""
    lo, hi = 0.0, 1.0
    while t_two_sided_p(hi, df) > alpha:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rejects_rows(rs, n: int, alpha: float = 0.05) -> np.ndarray:
    """Verwirft der t-Test (zweiseitig, Niveau alpha) die Nullhypothese ρ = 0, für jede Korrelation in rs? Gleichwertig zu t_test_p(r, n) < alpha."""
    rs = np.asarray(rs, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.abs(rs) * np.sqrt((n - 2) / (1.0 - rs * rs))
    return t > t_critical(n - 2, alpha)


def t_test_p(r: float, n: int) -> float:
    """Zweiseitiger p-Wert für H0: ρ = 0 (setzt Normalverteilung oder zumindest unabhängige Ränder voraus)."""
    t = t_statistic(r, n)
    return 0.0 if math.isinf(t) else t_two_sided_p(t, n - 2)


def fisher_interval(r: float, n: int, level: float = C.LEVEL) -> tuple[float, float]:
    """Fisher-z-Intervall: atanh(r) ± z/√(n−3), zurücktransformiert."""
    if n < 4:
        raise ValueError("das Fisher-Intervall braucht mindestens vier Beobachtungen")
    z = norm_ppf(0.5 + 0.5 * level)
    zr = math.atanh(max(-1.0 + 1e-12, min(1.0 - 1e-12, r)))
    se = 1.0 / math.sqrt(n - 3)
    return math.tanh(zr - z * se), math.tanh(zr + z * se)


def bootstrap_replicates(x, y, B: int, rng) -> np.ndarray:
    """B Bootstrap-Korrelationen: Beobachtungspaare mit Zurücklegen ziehen. Resamples mit konstanter Kennzahl ergeben NaN und bleiben erhalten."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    idx = rng.below(n, (B, n))
    xb, yb = x[idx], y[idx]
    xb = xb - xb.mean(1, keepdims=True)
    yb = yb - yb.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (xb * yb).sum(1) / np.sqrt((xb * xb).sum(1) * (yb * yb).sum(1))


def percentile_interval(rb, level: float = C.LEVEL) -> tuple[float, float]:
    """Perzentil-Intervall: die Quantile (1−level)/2 und (1+level)/2 der Bootstrap-Korrelationen (NaN ausgelassen)."""
    rb = np.asarray(rb, dtype=float)
    rb = rb[np.isfinite(rb)]
    if len(rb) == 0:
        raise ValueError("keine gültige Bootstrap-Korrelation")
    lo, hi = np.percentile(rb, [50.0 * (1.0 - level), 50.0 * (1.0 + level)])
    return float(lo), float(hi)


def bca_correction(rb, r_hat: float) -> float:
    """Verzerrungskorrektur z0 = Φ⁻¹(Anteil der Bootstrap-Werte unter r̂); der Anteil wird von 0 und 1 ferngehalten."""
    rb = np.asarray(rb, dtype=float)
    rb = rb[np.isfinite(rb)]
    B = len(rb)
    p = min(max(float((rb < r_hat).mean()), 0.5 / B), 1.0 - 0.5 / B)
    return norm_ppf(p)


def bca_acceleration(x, y) -> float:
    """Beschleunigung a = Σ(θ̄ − θ_i)³ / (6 (Σ(θ̄ − θ_i)²)^1,5) aus den Leave-one-out-Korrelationen; 0, wenn alle gleich sind."""
    th = jackknife_r(x, y)
    th = th[np.isfinite(th)]
    d = th.mean() - th
    den = 6.0 * float((d * d).sum()) ** 1.5
    return float((d**3).sum()) / den if den > 0.0 else 0.0


def bca_interval(x, y, rb, r_hat: float, level: float = C.LEVEL) -> tuple[float, float]:
    """BCa-Intervall: Perzentile an angepassten Stellen Φ(z0 + (z0 + z_q)/(1 − a(z0 + z_q)))."""
    rb = np.asarray(rb, dtype=float)
    rb = rb[np.isfinite(rb)]
    z0 = bca_correction(rb, r_hat)
    a = bca_acceleration(x, y)
    out = []
    for q in (0.5 * (1.0 - level), 0.5 * (1.0 + level)):
        zq = norm_ppf(q)
        out.append(norm_cdf(z0 + (z0 + zq) / (1.0 - a * (z0 + zq))))
    lo, hi = np.quantile(rb, out)
    return float(lo), float(hi)


def permutation_p(x, y, B: int, rng) -> float:
    """Zweiseitiger Permutationstest für Unabhängigkeit: (1 + #{|r*| ≥ |r|}) / (B + 1). Prüft Unabhängigkeit, nicht Unkorreliertheit."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    r = abs(pearson_r(x, y))
    perm = np.argsort(rng.uniform((B, n)), axis=1)
    yb = y[perm]
    xc = x - x.mean()
    yb = yb - yb.mean(1, keepdims=True)
    rs = np.abs((xc * yb).sum(1) / np.sqrt((xc * xc).sum() * (yb * yb).sum(1)))
    return float((1 + int((rs >= r - 1e-12).sum())) / (B + 1))


# --- Geschlossene Formeln --------------------------------------------------------------------------------------------------------------------

def variance_factor(nu: float) -> float:
    """1 + κ für die t-Verteilung mit ν Freiheitsgraden: E[w⁴]/E[w²]² = (ν − 2)/(ν − 4); unendlich für ν ≤ 4."""
    if math.isinf(nu):
        return 1.0
    return math.inf if nu <= 4 else (nu - 2.0) / (nu - 4.0)


def predicted_coverage(nu: float, level: float = C.LEVEL) -> float:
    """Erwartete Abdeckung des Fisher-Intervalls bei großem n: 2Φ(z/√(1+κ)) − 1; bei unendlicher Varianz geht sie gegen 0."""
    vf = variance_factor(nu)
    if math.isinf(vf):
        return 0.0
    return 2.0 * norm_cdf(norm_ppf(0.5 + 0.5 * level) / math.sqrt(vf)) - 1.0


def rmse_ratio(r: float) -> float:
    """RMSE der Regression geteilt durch die Streuung von y: √(1 − r²)."""
    return math.sqrt(max(0.0, 1.0 - r * r))


def sign_hit(r: float) -> float:
    """Anteil der Paare, in denen x und y dasselbe Vorzeichen haben (bivariate Normalverteilung): ½ + arcsin(r)/π."""
    return 0.5 + math.asin(r) / math.pi


def plan_n(rho: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Stichprobenumfang für einen zweiseitigen Test von ρ = 0 über Fisher-z: ((z_{1−α/2} + z_power)/atanh ρ)² + 3, aufgerundet."""
    if not 0.0 < abs(rho) < 1.0:
        raise ValueError("rho muss zwischen 0 und 1 (ausschließlich) liegen")
    return math.ceil(((norm_ppf(1.0 - alpha / 2.0) + norm_ppf(power)) / math.atanh(abs(rho))) ** 2 + 3.0)
