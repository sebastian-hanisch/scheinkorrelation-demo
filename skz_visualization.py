"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go

import skz_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", yanchor="top", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_series(x, y, height: int = 320):
    """Beide Reihen über die Wochen."""
    weeks = list(range(1, len(x) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weeks, y=x, mode="lines", name="x: Bestand Standort A", line=dict(color=C.COLOR_X, width=2), hovertemplate="Woche %{x}: %{y:.2f}<extra>x</extra>"))
    fig.add_trace(go.Scatter(x=weeks, y=y, mode="lines", name="y: Auftragseingang Standort B", line=dict(color=C.COLOR_Y, width=2), hovertemplate="Woche %{x}: %{y:.2f}<extra>y</extra>"))
    fig.update_xaxes(title_text="Woche")
    fig.update_yaxes(title_text="Wert der Reihe")
    return _base(fig, height)


def build_scatter(x, y, height: int = 320):
    """Streudiagramm der beiden Reihen mit Regressionsgerade; die Farbe läuft mit der Zeit (hell = früh, dunkel = spät)."""
    x, y = np.asarray(x), np.asarray(y)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Wochen", showlegend=False,
                             marker=dict(size=7, color=list(range(len(x))), colorscale="Blues", opacity=0.85, colorbar=dict(title="Woche", thickness=10)),
                             hovertemplate="x = %{x:.2f}, y = %{y:.2f}, Woche %{marker.color}<extra></extra>"))
    xc = x - x.mean()
    if float((xc * xc).sum()) > 0:
        slope = float((xc * (y - y.mean())).sum() / (xc * xc).sum())
        xs = np.array([x.min(), x.max()])
        fig.add_trace(go.Scatter(x=xs, y=y.mean() + slope * (xs - x.mean()), mode="lines", name="Regressionsgerade", line=dict(color=C.COLOR_TRUE, width=2, dash="dash"), hoverinfo="skip"))
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="y")
    return _base(fig, height)


def build_pvalues(results: dict, alpha: float = C.ALPHA, height: int = 280):
    """p-Werte der fünf Verfahren als Punkte auf logarithmischer Achse; orange = verwirft die Unabhängigkeit, grau = nicht."""
    keys = list(C.METHODS)
    names = [C.METHODS[k] for k in keys]
    vals = [max(results[k][1], 1e-4) if math.isfinite(results[k][1]) else 1.0 for k in keys]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vals, y=names, mode="markers+text", text=[f"{v:.3f}" if v >= 1e-3 else "< 0.001" for v in vals], textposition="middle right", showlegend=False,
                             marker=dict(size=13, color=[C.COLOR_FOUND if v <= alpha else C.COLOR_MISSED for v in vals]), hovertemplate="p = %{x:.4f}<extra>%{y}</extra>"))
    fig.add_vline(x=alpha, line=dict(color=C.COLOR_TRUE, dash="dash", width=1.5), annotation_text="5 %", annotation_position="top")
    ticks = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    fig.update_xaxes(title_text="p-Wert (logarithmisch)", type="log", range=[-4.1, math.log10(3.0)], tickvals=ticks, ticktext=[f"{t:g}" for t in ticks])
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=names, automargin=True)
    fig = _base(fig, height)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


def build_fa_lines(grid: dict, method: str, height: int = 360):
    """Fehlalarm (Anteil verworfener Unabhängigkeit bei unabhängigen Reihen) gegen T, eine Linie je Prozess, für ein Verfahren."""
    fig = go.Figure()
    for key in C.GRID_PROCESSES:
        rows = sorted((r for r in grid["fa"] if r["process"] == key), key=lambda r: r["T"])
        fig.add_trace(go.Scatter(x=[r["T"] for r in rows], y=[100 * r["rate"][method] for r in rows], mode="lines+markers", name=C.GRID_NAMES[key],
                                 line=dict(color=C.PROCESS_COLORS[key], width=3), hovertemplate="T = %{x}: %{y:.1f} %<extra>" + C.GRID_NAMES[key] + "</extra>"))
    fig.add_hline(y=5, line=dict(color=C.COLOR_TRUE, dash="dash", width=1.5))
    fig.update_xaxes(title_text="Länge T (Wochen)", type="log", tickvals=list(C.GRID_TS), ticktext=[str(t) for t in C.GRID_TS])
    fig.update_yaxes(title_text="Anteil „signifikant“ (%)", range=[0, 102])
    return _base(fig, height)


def build_r_histogram(sample, T: int, crit: float, height: int = 320):
    """Verteilung von r bei unabhängigen Paaren; die gestrichelten Linien sind die Schwellen ±r_krit des naiven Tests."""
    fig = go.Figure(go.Histogram(x=list(sample), xbins=dict(start=-1.0, end=1.0, size=0.05), marker_color=C.COLOR_X, hovertemplate="r um %{x}: %{y} Paare<extra></extra>"))
    for v in (-crit, crit):
        fig.add_vline(x=v, line=dict(color=C.COLOR_FOUND, dash="dash", width=2))
    fig.update_xaxes(title_text=f"Pearson-r der Reihen (T = {T})", range=[-1, 1])
    fig.update_yaxes(title_text="Zahl der Paare")
    return _base(fig, height)


def build_r_spread(grid: dict, height: int = 320):
    """Streuung von r gegen T für weißes Rauschen, AR(1) 0,9 und Irrfahrt; gestrichelt 1/√T."""
    fig = go.Figure()
    for key in ("weiss", "ar09", "walk"):
        rows = sorted((r for r in grid["fa"] if r["process"] == key), key=lambda r: r["T"])
        fig.add_trace(go.Scatter(x=[r["T"] for r in rows], y=[r["r"]["sd"] for r in rows], mode="lines+markers", name=C.GRID_NAMES[key], line=dict(color=C.PROCESS_COLORS[key], width=3),
                                 hovertemplate="T = %{x}: sd(r) = %{y:.3f}<extra>" + C.GRID_NAMES[key] + "</extra>"))
    ts = list(C.GRID_TS)
    fig.add_trace(go.Scatter(x=ts, y=[1 / math.sqrt(t) for t in ts], mode="lines", name="1/√T", line=dict(color=C.COLOR_TRUE, dash="dash", width=1.5), hoverinfo="skip"))
    fig.update_xaxes(title_text="Länge T (Wochen)", type="log", tickvals=ts, ticktext=[str(t) for t in ts])
    fig.update_yaxes(title_text="Streuung von r", range=[0, 0.55])
    return _base(fig, height)


def build_method_heatmap(table: dict, T: int, height: int = 380):
    """Fehlalarm je Prozess (Zeilen) und Verfahren (Spalten) für ein T, in Prozent."""
    keys = list(table)
    methods = list(C.METHODS)
    z = [[100 * table[k]["rate"][m] for m in methods] for k in keys]
    fig = go.Figure(go.Heatmap(z=z, x=[C.METHODS[m] for m in methods], y=[C.GRID_NAMES[k] for k in keys], zmin=0, zmax=100, colorscale="Oranges", showscale=False,
                               texttemplate="%{z:.0f}", textfont=dict(size=13), hovertemplate="%{y}, %{x}: %{z:.1f} %<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    return _base(fig, height)


def build_method_lines(grid: dict, key: str = "walk", height: int = 340):
    """Fehlalarm der fünf Verfahren gegen T für einen Prozess."""
    rows = sorted((r for r in grid["fa"] if r["process"] == key), key=lambda r: r["T"])
    fig = go.Figure()
    for m in C.METHODS:
        fig.add_trace(go.Scatter(x=[r["T"] for r in rows], y=[100 * r["rate"][m] for r in rows], mode="lines+markers", name=C.METHODS[m], line=dict(color=C.METHOD_COLORS[m], width=3),
                                 hovertemplate="T = %{x}: %{y:.1f} %<extra>" + C.METHODS[m] + "</extra>"))
    fig.add_trace(go.Scatter(x=[r["T"] for r in rows], y=[100 * r["rate"]["dw"] for r in rows], mode="lines+markers", name="Warnung R² > DW", line=dict(color=C.COLOR_TRUE, width=2, dash="dot"),
                             hovertemplate="T = %{x}: %{y:.1f} %<extra>Warnregel</extra>"))
    fig.add_hline(y=5, line=dict(color=C.COLOR_TRUE, dash="dash", width=1))
    fig.update_xaxes(title_text="Länge T (Wochen)", type="log", tickvals=list(C.GRID_TS), ticktext=[str(t) for t in C.GRID_TS])
    fig.update_yaxes(title_text="Anteil (%)", range=[0, 102])
    return _base(fig, height)


def build_relation(rows: list[dict], scenario: str, T: int, height: int = 340):
    """Anteil verworfener Unabhängigkeit bei echtem Zusammenhang: je ρ eine Balkengruppe, je Verfahren ein Balken."""
    sub = sorted((r for r in rows if r["scenario"] == scenario and r["T"] == T), key=lambda r: r["rho"])
    fig = go.Figure()
    labels = ["y = x + Rauschen"] if scenario == "coint" else [f"ρ = {r['rho']:g}" for r in sub]
    for m in C.METHODS:
        fig.add_trace(go.Bar(x=labels, y=[100 * r["rate"][m] for r in sub], name=C.METHODS[m], marker_color=C.METHOD_COLORS[m]))
    fig.add_hline(y=5, line=dict(color=C.COLOR_TRUE, dash="dash", width=1.5))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Anteil verworfener Unabhängigkeit (%)", range=[0, 102])
    return _base(fig, height)


def build_variance(rows: list[dict], height: int = 320):
    """T · Var(r) gegen φ (Simulation: Punkte je T; Formel (1 + φ²)/(1 − φ²): Linie), logarithmische y-Achse."""
    fig = go.Figure()
    phis = sorted({r["phi"] for r in rows})
    fine = np.linspace(min(phis), max(phis), 100)
    fig.add_trace(go.Scatter(x=fine, y=(1 + fine**2) / (1 - fine**2), mode="lines", name="Formel (1 + φ²)/(1 − φ²)", line=dict(color=C.COLOR_TRUE, dash="dash", width=2), hoverinfo="skip"))
    for T, color in zip(sorted({r["T"] for r in rows}), ("#9ecae1", "#08519c")):
        sub = sorted((r for r in rows if r["T"] == T), key=lambda r: r["phi"])
        fig.add_trace(go.Scatter(x=[r["phi"] for r in sub], y=[r["t_var"] for r in sub], mode="lines+markers", name=f"Simulation, T = {T}", line=dict(color=color, width=3),
                                 hovertemplate="φ = %{x}: T·Var(r) = %{y:.2f}<extra>T = " + str(T) + "</extra>"))
    fig.update_xaxes(title_text="φ der beiden AR(1)-Reihen")
    fig.update_yaxes(title_text="T · Var(r)", type="log")
    return _base(fig, height)
