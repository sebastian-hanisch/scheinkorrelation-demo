"""Scheinkorrelation: wenn zwei Reihen ohne Zusammenhang „signifikant“ korrelieren - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Viertes Stück der Korrelations-Reihe. Zwei Wochenreihen zweier Standorte (Bestand an A, Auftragseingang an B), unabhängig simuliert: Wie oft findet der naive Test trotzdem einen
„signifikanten“ Zusammenhang, und welches Gegenmittel (Differenzieren, Trendbereinigung, effektive Stichprobengröße, Verschiebungstest, Warnregel) hilft wann, zu welchem Preis?

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import skz_constants as C
import skz_evaluation as ev
from skz_layout import dunkles_design, kopf_css
from skz_pdf_export import build_pdf
from skz_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from skz_visualization import (
    build_fa_lines,
    build_method_heatmap,
    build_method_lines,
    build_pvalues,
    build_r_histogram,
    build_r_spread,
    build_relation,
    build_scatter,
    build_series,
    build_variance,
)

st.set_page_config(page_title="Scheinkorrelation – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _grid():
    return ev.load_grid()


@st.cache_data(show_spinner=False)
def _sample(key, T):
    return ev.r_sample(key, T)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in skz_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📈 Scheinkorrelation: wenn zwei Reihen ohne Zusammenhang „signifikant“ korrelieren")
st.markdown(
    """
**Viertes Stück der Korrelations-Reihe** und **die erste Demo des Portfolios mit Zeitreihen im Korrelations-Thema**. Zwei **Wochenreihen** zweier Standorte: der **Bestand** an Standort A (x) und der
**Auftragseingang** an Standort B (y). Sie sind **unabhängig simuliert**, die Wahrheit ist bekannt: es gibt keinen Zusammenhang. Die ersten drei Stücke nahmen unabhängige Beobachtungen an. Hier hängt jede
Woche an der vorigen, und der Test auf „r = 0“ bemerkt das nicht. **Wie oft findet er trotzdem einen Zusammenhang, und welches Gegenmittel hilft wann, zu welchem Preis?**

Weiter unten: fünf Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**. Das Intervall für einen **Mittelwert** bei Autokorrelation steht in `output-analysis-demo`,
Prognosen mit Differenzen in `arima-demo`, der echte Gleichlauf (Kointegration) im Stück dazu, Ursachen in der Linie *Kausale Diagramme und Adjustierung*.
"""
)

with st.expander("So entsteht die Scheinkorrelation", expanded=True):
    st.markdown(
        """
Eine Reihe mit **Gedächtnis** (jede Woche hängt an der vorigen) driftet: sie bleibt lange oberhalb oder unterhalb ihres Mittels. Zwei solche Reihen laufen dann zufällig eine Weile in dieselbe Richtung,
und r misst genau das. Der naive Test rechnet, als wären die Wochen unabhängig (T − 2 Freiheitsgrade), und ist deshalb zu gutgläubig.

| Prozess | Gedächtnis | Wirkung auf r |
|---|---|---|
| **weißes Rauschen** | keines | Kontrolle: der Test hält seine 5 % |
| **AR(1)** mit φ | Rückkehr zum Mittel, langsam bei φ nahe 1 | r streut mehr, als der Test annimmt |
| **Irrfahrt** | unbegrenzt (φ = 1) | r schrumpft **nicht** mit der Länge der Reihe |
| **Trend plus Rauschen** | gemeinsame Richtung | r wird mit T immer größer |

| Verfahren | Idee | hilft bei | versagt bei |
|---|---|---|---|
| **naiver Test** | t-Test von r mit T − 2 Freiheitsgraden | weißes Rauschen | alles mit Gedächtnis oder Trend |
| **Differenzieren** | Korrelation der Wochenänderungen | Irrfahrt, AR(1) | weißes Rauschen, Trend (Überdifferenzieren) |
| **Trend weg** | Trendgerade abziehen | Trend plus Rauschen | Irrfahrt, AR(1) |
| **n_eff** | weniger „freie“ Wochen aus den Autokorrelationen | Rauschen, AR(1), Irrfahrt | Trend; bei echtem Zusammenhang in den Zufallsstößen zu vorsichtig |
| **Verschiebung** | y zyklisch verschieben, r jeweils neu | AR(1) bis φ = 0,9 | φ = 0,99, Irrfahrt, Trend |
| **Warnregel R² > DW** | Residuen der Regression stark autokorreliert | Irrfahrt | Trend (schweigt) |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Reihen**")
    process = st.selectbox("Prozess", list(C.PROCESSES), format_func=C.PROCESSES.get, key="process_select",
                           help="Wie jede der beiden Reihen entsteht. Beide Reihen haben denselben Prozess, aber eigene Zufallszahlen.")
    phi = st.slider("φ (nur bei AR(1))", *bounds("phi_slider"), step=0.01, key="phi_slider", help="Wie stark jede Woche an der vorigen hängt. Wirkt nur beim Prozess AR(1).")
    slope = st.slider("Trend je Woche (nur bei Trend)", *bounds("slope_slider"), step=0.01, key="slope_slider", help="Steigung der gemeinsamen Trendgeraden in Einheiten der Streuung des Rauschens. Wirkt nur beim Prozess Trend plus Rauschen.")
    T = st.slider("Länge T (Wochen)", *bounds("T_slider"), key="T_slider", help="Zahl der Wochen. 104 Wochen sind zwei Jahre.")
    rho = st.slider("Zusammenhang ρ der Innovationen", *bounds("rho_slider"), step=0.05, key="rho_slider",
                    help="0 heißt unabhängig. Ab 0,05 sind die Zufallsstöße beider Reihen mit der Korrelation ρ verbunden, es gibt einen echten Zusammenhang.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Paar ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"process_select": process, "phi_slider": round(float(phi), 2), "slope_slider": round(float(slope), 2), "T_slider": int(T), "rho_slider": round(float(rho), 2), "seed_input": int(seed)})

settings = ev.Settings(process, round(float(phi), 2), round(float(slope), 2), int(T), round(float(rho), 2), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
r_naive, p_naive = a.results["naiv"]
cols = st.columns(5)
cols[0].metric("Korrelation r", f"{r_naive:.2f}", help="Pearson-Korrelation der beiden Reihen über alle Wochen.")
cols[1].metric("p-Wert (naiv)", f"{p_naive:.3f}", help="Zweiseitiger t-Test von r mit T − 2 Freiheitsgraden: unter 0,05 gilt der Zusammenhang als „signifikant“.")
cols[2].metric("R² der Regression", f"{a.r2:.2f}", help="Quadrat von r.")
cols[3].metric("Durbin–Watson", f"{a.dw:.2f}", help="Nahe 2: Residuen ohne Autokorrelation; nahe 0: stark autokorrelierte Residuen.")
cols[4].metric("Warnregel R² > DW", "Warnung" if a.warning else "keine", help="Granger–Newbold: ist R² größer als DW, ist die Regression verdächtig (Scheinregression).")
left, right = st.columns(2)
with left:
    st.plotly_chart(build_series(a.x, a.y), width="stretch", key="main_series")
    st.caption("Beide Reihen über die Wochen: bei Irrfahrten laufen sie oft eine lange Strecke gemeinsam hoch oder runter, obwohl sie nichts miteinander zu tun haben.")
with right:
    st.plotly_chart(build_scatter(a.x, a.y), width="stretch", key="main_scatter")
    st.caption("Dieselben Wochen als Streudiagramm; die Farbe läuft mit der Zeit. Ein „Zusammenhang“ entsteht oft aus aufeinanderfolgenden Wochen.")
st.plotly_chart(build_pvalues(a.results), width="stretch", key="main_pvalues")
st.caption("p-Werte der fünf Verfahren auf demselben Paar: orange = verwirft die Unabhängigkeit (p ≤ 5 %), grau = nicht. Gestrichelt: 5 %.")
if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = (settings, build_pdf(a))
pdf = st.session_state.get("pdf_bytes")
if pdf and pdf[0] == settings:
    st.download_button("⬇️ PDF herunterladen", pdf[1], file_name="scheinkorrelation-demo.pdf", mime="application/pdf", key="pdf_download")

st.markdown("## ❓ Kernfrage: Ist da etwas, oder sieht es nur so aus?")
kind, text = ev.verdict_main(a)
getattr(st, kind)(text)

st.markdown("---")

grid = _grid()
if grid is None:
    st.warning("Das vorgerechnete Raster (data/spurious_grid.json) fehlt; es entsteht mit `python tools/make_grid.py`. Die Experimente 1 bis 4 zeigen es.")

# --- E1 Wie oft? ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wie oft findet ein Test einen Zusammenhang, den es nicht gibt?")
st.caption("Anteil der Läufe, in denen das Verfahren bei **unabhängigen** Reihen die Unabhängigkeit verwirft (5 %-Niveau, erwartet 5 %). Vorgerechnet: 4 Seeds mit je 1 000 Paaren je Prozess und Länge.")
if grid is not None:
    e1c1, e1c2 = st.columns(2)
    e1_method = e1c1.selectbox("Verfahren", list(C.METHODS), format_func=C.METHODS.get, key="e1_method")
    e1_T = e1c2.selectbox("T für das Verdikt", list(C.GRID_TS), index=2, key="e1_T")
    st.plotly_chart(build_fa_lines(grid, e1_method), width="stretch", key="fa_lines")
    kind, text = ev.verdict_fa(ev.fa_table(grid, e1_T), e1_T)
    getattr(st, kind)(text)
    st.caption("Beim naiven Test steigt der Fehlalarm der Irrfahrt mit der Länge der Reihe, statt zu sinken: mehr Wochen machen die Scheinkorrelation nicht seltener.")

st.markdown("---")

# --- E2 Wie groß wird r? -------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Wie groß wird r, wenn nichts da ist?")
if grid is not None:
    st.plotly_chart(build_r_spread(grid), width="stretch", key="r_spread")
    st.caption("Streuung von r bei unabhängigen Reihen gegen die Länge. Bei weißem Rauschen schrumpft sie wie 1/√T (gestrichelt); bei der Irrfahrt bleibt sie bei rund 0,5.")
    e2c1, e2c2 = st.columns(2)
    e2_process = e2c1.selectbox("Prozess", list(C.GRID_PROCESSES), format_func=C.GRID_NAMES.get, index=4, key="e2_process")
    e2_T = e2c2.selectbox("T", list(C.GRID_TS), index=2, key="e2_T")
    kind, text = ev.verdict_r(ev.find(grid["fa"], process=e2_process, T=e2_T))
    getattr(st, kind)(text)
    sample = _sample(e2_process, e2_T)
    st.plotly_chart(build_r_histogram(sample, e2_T, ev.r_critical(e2_T)), width="stretch", key="r_hist")
    kind, text = ev.verdict_hist(sample, e2_T)
    getattr(st, kind)(text)
    st.caption("2 000 neu gezogene Paare (live). Die orangen Linien sind die Schwellen ±r des naiven Tests: alles außerhalb gilt als „signifikant“.")

st.markdown("---")

# --- E3 Gegenmittel ------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Die Gegenmittel und ihr Preis")
st.caption("Fehlalarm bei **unabhängigen** Reihen je Prozess und Verfahren in Prozent (erwartet 5). Orange: zu oft, auch zu oft bei Differenzieren von weißem Rauschen und Trend.")
if grid is not None:
    e3_T = st.selectbox("T", list(C.GRID_TS), index=2, key="e3_T")
    table = ev.fa_table(grid, e3_T)
    st.plotly_chart(build_method_heatmap(table, e3_T), width="stretch", key="method_heatmap")
    kind, text = ev.verdict_methods(table, e3_T)
    getattr(st, kind)(text)
    kind, text = ev.verdict_overdiff(table, e3_T)
    getattr(st, kind)(text)
    e3_process = st.selectbox("Prozess für den Verlauf gegen T", list(C.GRID_PROCESSES), format_func=C.GRID_NAMES.get, index=4, key="e3_process")
    st.plotly_chart(build_method_lines(grid, e3_process), width="stretch", key="method_lines")
    kind, text = ev.verdict_warning(grid)
    getattr(st, kind)(text)
    st.caption("Die Warnregel R² > DW ist kein Test, sondern ein Hinweis; sie wird mit der Länge der Reihe treffsicherer, schweigt aber beim Trend.")

st.markdown("---")

# --- E4 Echter Zusammenhang ----------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Und wenn es den Zusammenhang wirklich gibt?")
st.caption("Anteil verworfener Unabhängigkeit bei **echtem** Zusammenhang (Innovationen mit Korrelation ρ). Ein Gegenmittel darf einen echten Zusammenhang nicht verdecken; der naive Test kann beide Welten nicht unterscheiden.")
if grid is not None:
    e4c1, e4c2 = st.columns(2)
    e4_scenario = e4c1.selectbox("Szenario", list(C.REL_SCENARIOS), format_func=C.REL_SCENARIOS.get, key="e4_scenario")
    e4_T = e4c2.selectbox("T", list(C.REL_TS), index=1, key="e4_T")
    st.plotly_chart(build_relation(grid["rel"], e4_scenario, e4_T), width="stretch", key="relation_chart")
    kind, text = ev.verdict_relation(grid["rel"], e4_scenario, e4_T)
    getattr(st, kind)(text)
    st.caption("Bei Irrfahrten mit ρ = 0,3 verwirft der naive Test fast so oft wie bei ρ = 0: er sieht den Unterschied nicht. Differenzieren schon, n_eff ist dafür zu vorsichtig.")

st.markdown("---")

# --- E5 Varianz von r ----------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Warum? Die Varianz von r bei AR(1)")
st.markdown(
    "Bei unabhängigen AR(1)-Reihen mit gleichem φ gilt für lange Reihen **T · Var(r) = (1 + φ²)/(1 − φ²)**: der Test müsste mit der **effektiven Stichprobengröße** T · (1 − φ²)/(1 + φ²) rechnen "
    "statt mit T. Hier gegen die Simulation (4 · 250 Läufe je Zeile)."
)
if st.button("🔁 Varianz rechnen (wenige Sekunden)", key="btn_var"):
    with st.spinner("Rechne die Varianzen …"):
        st.session_state["var_result"] = [ev.variance_cell(f, t) for t in C.VAR_TS for f in C.VAR_PHIS]
var = st.session_state.get("var_result")
if var:
    st.plotly_chart(build_variance(var), width="stretch", key="var_chart")
    tab = ["| φ | T | T · Var(r) | Formel | effektive Stichprobengröße |", "|---|---|---|---|---|"]
    tab += [f"| {r['phi']:g} | {r['T']} | {r['t_var']:.2f} | {r['formula']:.2f} | {r['n_eff']:.1f} |" for r in sorted(var, key=lambda r: (r["phi"], r["T"]))]
    st.markdown("\n".join(tab))
    kind, text = ev.verdict_variance(var)
    getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die vier Prozesse stehen für die Wirklichkeit** | es sind Beispiele; Saison, Brüche und sehr lange Gedächtnisse zeigt die Demo nicht | – |
| **Ein bestandener Test beweist einen Zusammenhang** | bei Gedächtnis oder Trend ist „signifikant“ oft nur gemeinsames Driften (Experiment 1) | Gegenmittel und Domänenwissen |
| **Differenzieren schadet nie** | bei weißem Rauschen und bei Trend plus Rauschen hebt es den Fehlalarm (Experiment 3) | Prozess vorher prüfen |
| **n_eff genügt** | die geschätzte Korrektur versagt beim Trend und ist bei Irrfahrten mit korrelierten Zuwächsen zu vorsichtig (Experimente 3 und 4); bei y = x + Rauschen findet sie den Zusammenhang | – |
| **Das Verschiebungstest ist verteilungsfrei und deshalb sicher** | es hält nur bei stationären Reihen mit kurzem Gedächtnis (Experiment 3) | – |
| **Die Warnregel erkennt jede Scheinregression** | sie schweigt beim Trend und warnt bei kurzen Reihen selten (Experiment 3) | – |
| **Echter Gleichlauf** | ob zwei Irrfahrten *gemeinsam* driften (Kointegration), prüft kein Verfahren hier | Stück zu Kointegration und gerichteter Abhängigkeit |
| **Beobachtungen sind unabhängig** | das ist die Annahme, die diese Demo verletzt | – |
| **Ursachen** | auch ein echter Zusammenhang sagt nicht, welche Reihe die andere bewegt | Linie *Kausale Diagramme und Adjustierung* |
"""
)
st.caption("Alle Reihen sind simuliert; die Prozesse sind Modellwahlen, kein Befund über echte Standorte. Die effektive Stichprobengröße ist hier die Näherung mit den Lag-1-Autokorrelationen beider Reihen.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind zwei Reihen $x_1,\dots,x_T$ und $y_1,\dots,y_T$; gefragt ist, ob sie unabhängig sind.

| Symbol | Bedeutung |
|---|---|
| $r$ | Pearson-Korrelation der beiden Reihen |
| $\varepsilon_t$ | standardnormale Zufallsstöße (Innovationen) |
| $\varphi$ | Gedächtnis des AR(1) |
| $\rho$ | Korrelation der Innovationen beider Reihen (0 = unabhängig) |
| $r_{1,x}, r_{1,y}$ | Autokorrelation zum Lag 1 der beiden Reihen |

**Prozesse.**

$$ x_t = \varepsilon_t \ \ (\text{weißes Rauschen}), \qquad x_t = \varphi\, x_{t-1} + \varepsilon_t \ \ (\text{AR}(1)), \qquad x_t = x_{t-1} + \varepsilon_t \ \ (\text{Irrfahrt}), \qquad x_t = b\,t + \varepsilon_t \ \ (\text{Trend}) $$

**Naiver Test.**

$$ t = r\sqrt{\frac{T-2}{1-r^2}} \sim t_{T-2} \ \text{ (nur bei unabhängigen Beobachtungen)} $$

**Varianz von $r$** bei unabhängigen AR(1)-Reihen gleichen $\varphi$ und langem $T$, und die daraus folgende effektive Stichprobengröße:

$$ T\cdot\operatorname{Var}(r) \to \frac{1+\varphi^2}{1-\varphi^2}, \qquad T_{\mathrm{eff}} = T\,\frac{1-\varphi^2}{1+\varphi^2} $$

**Schätzung mit den Lag-1-Autokorrelationen** (nach oben durch $T$ begrenzt, mindestens 3) und der $t$-Test mit $T_{\mathrm{eff}}-2$ Freiheitsgraden:

$$ T_{\mathrm{eff}} = T\,\frac{1 - r_{1,x}\,r_{1,y}}{1 + r_{1,x}\,r_{1,y}} $$

**Differenzieren und Trendbereinigung.**

$$ \Delta x_t = x_t - x_{t-1}, \qquad \tilde x_t = x_t - \hat a - \hat b\, t \quad (\text{Residuen der Regression auf die Zeit}) $$

**Zirkuläres Verschiebungstest.** Mit der um $s$ Wochen verschobenen Reihe $y^{(s)}_t = y_{(t+s) \bmod T}$:

$$ p = \frac{1}{T}\,\#\bigl\{ s \in \{0,\dots,T-1\} : \lvert r(x, y^{(s)})\rvert \ge \lvert r(x,y)\rvert \bigr\} $$

**Warnregel (Granger–Newbold).** Mit den Residuen $u_t$ der Regression $y \sim x$:

$$ \mathrm{DW} = \frac{\sum_{t=2}^{T}(u_t - u_{t-1})^2}{\sum_{t=1}^{T} u_t^2}, \qquad \text{Warnung, wenn } R^2 = r^2 > \mathrm{DW} $$

**Quellen.** In dieser Fassung sind keine Literaturangaben ausgewiesen; die Formeln sind durch unabhängige Rechenwege (Tests) und die Messungen der Experimente gestützt.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
