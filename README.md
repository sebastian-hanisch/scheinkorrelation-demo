# Scheinkorrelation: wenn zwei Reihen ohne Zusammenhang „signifikant“ korrelieren – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-scheinkorrelation-demo.streamlit.app/)** -->

Viertes Stück der Korrelations-Reihe in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Nachfolger von [pearson-demo](https://github.com/sebastian-hanisch/pearson-demo) (Kante `fix`: die Beobachtungen sind nicht unabhängig), neben [rangkorrelation-demo](https://github.com/sebastian-hanisch/rangkorrelation-demo) und [abhaengigkeitsmasse-demo](https://github.com/sebastian-hanisch/abhaengigkeitsmasse-demo). Es ist die **erste Demo des Portfolios mit Zeitreihen im Korrelations-Thema**: zwei **Wochenreihen** zweier Standorte (x: Bestand an Standort A, y: Auftragseingang an Standort B), **unabhängig simuliert**, die Wahrheit ist bekannt: es gibt keinen Zusammenhang. Die Fragen: **Wie oft findet der naive Test trotzdem einen „signifikanten“, und welches Gegenmittel hilft wann, zu welchem Preis?**

## Warum dieses Problem

Der t-Test auf r = 0 rechnet, als wären die Wochen unabhängig. Bei Reihen mit Gedächtnis ist das falsch, und die Folge ist drastisch. Befunde: **zwei unabhängige Irrfahrten sind bei 104 Wochen in 76,1 % der Läufe „signifikant“** (89,4 % bei 520 Wochen), und die Streuung von r **schrumpft nicht mit der Länge** (0,489 bei 26, 0,484 bei 520 Wochen; bei weißem Rauschen 0,200 und 0,044). **Differenzieren** heilt die Irrfahrt (4,4 %), hebt aber den Fehlalarm bei weißem Rauschen auf **11,3 %** und bei Trend plus Rauschen auf **10,3 %**. **Trendbereinigung** heilt nur den Trend (5,0 %), die Irrfahrt bleibt bei 64,8 %. Die **effektive Stichprobengröße** mit geschätzten Autokorrelationen hält 4,0 bis 5,1 % bei Rauschen, AR(1) und Irrfahrt, **versagt aber beim Trend** (100,0 %) und ist bei echtem Zusammenhang zu vorsichtig (8,6 % statt 88,2 % beim Differenzieren). Die **Warnregel R² > DW** warnt bei unabhängigen Irrfahrten mit wachsender Länge öfter (10,3 % bei 26, 81,5 % bei 520 Wochen) und schweigt beim Trend.

## Befunde und Korrekturen gegenüber dem Plan

Der Einzelplan (`korrelation-planung/STUECK-4-scheinkorrelation.md`, nicht in diesem Repo) stützte sich auf Wegwerfskripte mit gleichem Kern, anderer Ziehungsreihenfolge und teils anderen Läufen. Gebaut ist alles mit SplitMix64 und eigenem Kern; die Zahlen weichen leicht ab, die Befunde bleiben:

- **Irrfahrt, T = 104:** naiver Test 76,1 % (Plan: 76,8 %), Differenzieren 4,4 % (5,2 %), n_eff 4,6 % (4,7 %), Verschiebung 22,0 % (22,2 %), Warnregel 55,9 % (55,7 %).
- **Überdifferenzieren, T = 104:** weißes Rauschen 11,3 % (Plan: 10,3 %), Trend plus Rauschen 10,3 % (10,7 %); über alle fünf Längen 10,4 bis 11,8 % bzw. 10,2 bis 10,8 %.
- **AR(1) 0,9, T = 104:** naiver Test 50,4 % (Plan: 51,0 %).
- **Echter Zusammenhang, Irrfahrten ρ = 0,3, T = 104:** naiver Test 79,4 % (Plan: 79,7 %), Differenzieren 88,2 % (87,8 %), n_eff 8,6 % (7,7 %).
- **Neu gegenüber dem Plan:** die E2-Verteilung von r wird zusätzlich live gezogen (2 000 Paare mit den Schwellen des naiven Tests); die Streuung über die Seeds steht für jede Zelle im Raster.

## Modell

Zwei Reihen der Länge T (Wochen) mit **demselben Prozess**, aber eigenen Zufallszahlen (Erzeuger `skz_algorithm.py`, Zufall aus SplitMix64 in wörtlichen Kopien von `pearson-demo`: `prs_rng.py`, `prs_constants.py`, `prs_scenario.py`, `prs_algorithm.py`): **weißes Rauschen** x_t = ε_t; **AR(1)** x_t = φ x_{t−1} + ε_t mit stationärem Start; **Irrfahrt** x_t = x_{t−1} + ε_t (x₀ = ε₀); **Trend plus Rauschen** x_t = b·t + ε_t (b = 0,05 je Woche im Raster). Die Innovationen beider Reihen haben die Korrelation ρ (0: unabhängig); dazu der Sonderfall **gemeinsame Irrfahrt** y = x + ε. Der Regler „Prozess“ hat vier Einträge, das Raster sechs (AR(1) mit φ = 0,5 / 0,9 / 0,99).

## Methodik

- **Fünf Verfahren** in `skz_algorithm.py`, ohne scipy zur Laufzeit, alle auf demselben Paar: **naiver t-Test** (T − 2 Freiheitsgrade), **Test auf die Differenzen** (T − 1 Beobachtungen), **Test nach Abzug der Trendgeraden**, **t-Test mit effektiver Stichprobengröße** n_eff = T (1 − r₁ₓ r₁ᵧ)/(1 + r₁ₓ r₁ᵧ) (höchstens T, mindestens 3) und **zirkuläres Verschiebungstest** über alle T Verschiebungen per FFT. Dazu die **Warnregel R² > Durbin–Watson** (Granger–Newbold) als Hinweis, nicht als Test.
- **Experimente:** (1) Fehlalarm je Prozess und Länge, (2) Verteilung von r (Raster und live), (3) Gegenmittel und ihr Preis, (4) echter Zusammenhang, (5) Varianz von r bei AR(1) gegen die Formel (live klein).
- **Vorgerechnetes Raster** (`tools/make_grid.py`, `data/spurious_grid.json`, 3 Sekunden auf 10 Prozessen): Fehlalarm und Verteilung von r (30 Zellen: 6 Prozesse · T = 26, 52, 104, 260, 520, je 4 · 1 000 Paare), echter Zusammenhang (30 Zellen: Irrfahrten, AR(1) 0,9 und Trend mit ρ = 0, 0,3, 0,6 sowie die gemeinsame Irrfahrt, T = 52, 104, 260). Vier Seeds je Zelle (101 bis 104), die Streuung über die Seeds steht im Raster. **Live:** die Verteilung von r (2 000 Paare) und die Varianz (4 · 250 Läufe je φ).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (Raster-Werte gegen die JSON-Datei, Live-Werte über dieselben Funktionen wie in der App). Anteile verworfener Unabhängigkeit bei 5 % Niveau, in Prozent der 4 000 Läufe je Zelle; die Streuung über die Seeds liegt in jeder Zelle bei höchstens 2,6 Punkten.

**Fehlalarm des naiven Tests bei unabhängigen Reihen** (Experiment 1):

| Prozess | T = 26 | 52 | 104 | 260 | 520 |
|---|---|---|---|---|---|
| weißes Rauschen | 5,0 | 4,9 | 4,8 | 4,9 | 5,0 |
| AR(1) 0,5 | 11,2 | 13,0 | 12,4 | 12,7 | 12,6 |
| AR(1) 0,9 | 38,9 | 47,6 | 50,4 | 51,3 | 53,0 |
| AR(1) 0,99 | 52,3 | 63,2 | 74,1 | 79,7 | 83,0 |
| Irrfahrt | 52,5 | 67,7 | 76,1 | 86,5 | 89,4 |
| Trend + Rauschen | 8,9 | 79,3 | 100,0 | 100,0 | 100,0 |

**Gegenmittel bei unabhängigen Reihen, T = 104** (Experiment 3; Anteil verworfen bzw. Anteil mit R² > DW):

| Prozess | naiv | Differenzieren | Trend weg | n_eff | Verschiebung | R² > DW |
|---|---|---|---|---|---|---|
| weißes Rauschen | 4,8 | 11,3 | 5,1 | 4,8 | 4,9 | 0,0 |
| AR(1) 0,5 | 12,4 | 7,8 | 12,4 | 5,1 | 5,0 | 0,0 |
| AR(1) 0,9 | 50,4 | 5,1 | 47,3 | 4,7 | 5,5 | 1,8 |
| AR(1) 0,99 | 74,1 | 5,0 | 65,2 | 4,0 | 16,0 | 48,9 |
| Irrfahrt | 76,1 | 4,4 | 64,8 | 4,6 | 22,0 | 55,9 |
| Trend + Rauschen | 100,0 | 10,3 | 5,0 | 100,0 | 99,2 | 0,0 |

**Echter Zusammenhang, T = 104** (Experiment 4; Innovationen mit Korrelation ρ, Anteil verworfen):

| Szenario | ρ | naiv | Differenzieren | Trend weg | n_eff | Verschiebung | R² > DW |
|---|---|---|---|---|---|---|---|
| zwei Irrfahrten, Zuwächse korreliert | 0 | 76,1 | 5,2 | 66,9 | 5,0 | 22,0 | 54,6 |
| | 0,3 | 79,4 | 88,2 | 74,2 | 8,6 | 29,3 | 61,1 |
| | 0,6 | 89,0 | 100,0 | 91,7 | 26,8 | 54,4 | 78,1 |
| zwei AR(1) 0,9, Innovationen korreliert | 0 | 51,2 | 5,2 | 48,9 | 4,7 | 5,2 | 1,9 |
| | 0,3 | 72,0 | 87,9 | 71,2 | 20,4 | 26,2 | 13,9 |
| | 0,6 | 96,7 | 100,0 | 97,3 | 73,2 | 77,0 | 65,0 |
| gemeinsamer Trend, Rauschen korreliert | 0 | 100,0 | 10,6 | 5,0 | 100,0 | 99,6 | 0,0 |
| | 0,3 | 100,0 | 82,7 | 87,9 | 100,0 | 100,0 | 0,0 |
| | 0,6 | 100,0 | 100,0 | 100,0 | 100,0 | 100,0 | 0,0 |
| gemeinsame Irrfahrt, y = x + Rauschen | – | 100,0 | 100,0 | 100,0 | 98,0 | 100,0 | 0,0 |

| Frage | Befund |
|---|---|
| Wie oft täuscht der naive Test? (Experiment 1) | Bei **unabhängigen Irrfahrten** steigt der Fehlalarm mit der Länge: 52,5 / 67,7 / 76,1 / 86,5 / 89,4 % bei T = 26 / 52 / 104 / 260 / 520; bei weißem Rauschen bleibt er bei 4,8 bis 5,0 %. AR(1) 0,9 sättigt bei rund 51 % (53,0 % bei T = 520), AR(1) 0,99 steigt bis 83,0 %. **Trend plus Rauschen** kippt zwischen T = 26 (8,9 %) und T = 104 (100,0 %). |
| Wie groß wird r? (Experiment 2) | **Bei der Irrfahrt schrumpft r nicht mit T:** Streuung von r 0,489 / 0,493 / 0,488 / 0,484 / 0,484, mittleres \|r\| 0,417 bis 0,424, in 38,4 bis 40,4 % der Paare ist \|r\| > 0,5, auch bei 520 Wochen. Bei weißem Rauschen sinkt die Streuung wie 1/√T: 0,200 / 0,139 / 0,099 / 0,062 / 0,044. Live (2 000 Paare, T = 104): bei der Irrfahrt liegt \|r\| in 76,7 % der Paare über der Schwelle 0,193 des naiven Tests, bei weißem Rauschen in 5,4 %. |
| Hilft Differenzieren? (Experiment 3) | **Bei Irrfahrten und AR(1) ab 0,9: ja** (4,4 bis 5,1 % bei T = 104, über alle Längen 4,4 bis 5,6 %); **bei weißem Rauschen und bei Trend plus Rauschen: nein, Überdifferenzieren** hebt den Fehlalarm auf 10,4 bis 11,8 % bzw. 10,2 bis 10,8 % (alle fünf Längen); AR(1) 0,5: 6,7 bis 7,8 %. |
| Hilft Trendbereinigung? | **Nur beim Trend:** 5,0 % bei T = 104; die Irrfahrt bleibt bei 64,8 % (T = 104) und 84,2 % bei T = 520, AR(1) 0,9 bei 47,3 %. |
| Hilft n_eff? | **Bei Rauschen, AR(1) und Irrfahrt ja** (4,0 bis 5,1 % bei T = 104), **beim Trend nein** (100,0 %), und bei echtem Zusammenhang **zu vorsichtig:** bei Irrfahrten mit ρ = 0,3 nur 8,6 %, bei AR(1) 0,9 nur 20,4 % (Differenzieren: 88,2 und 87,9 %). |
| Hilft das Verschiebungstest? | **Nur bei kurzem Gedächtnis:** 4,9 bis 5,5 % bei Rauschen, AR(1) 0,5 und 0,9; bei AR(1) 0,99 16,0 %, bei der Irrfahrt 22,0 %, beim Trend 99,2 %. |
| Erkennt die Warnregel die Scheinregression? | **Mit wachsender Länge besser, beim Trend gar nicht:** bei unabhängigen Irrfahrten warnt R² > DW in 10,3 / 34,3 / 55,9 / 74,1 / 81,5 % der Läufe (T = 26 bis 520), bei Trend plus Rauschen und bei weißem Rauschen nie (0,0 %); auch bei echter Niveaubeziehung y = x + Rauschen nicht (0,0 %). |
| Sieht der naive Test den Unterschied? (Experiment 4) | **Nein:** bei zwei Irrfahrten verwirft er ohne Zusammenhang in 76,1 %, mit ρ = 0,3 in 79,4 %; beim gemeinsamen Trend in beiden Fällen 100,0 %. Differenzieren trennt: 5,2 % gegen 88,2 % bei Irrfahrten, 5,2 % gegen 87,9 % bei AR(1) 0,9. |
| Stimmt die Varianzformel? (Experiment 5, live) | T · Var(r) gegen (1 + φ²)/(1 − φ²) für T = 104 / 520: φ = 0: 1,02 / 0,98 (Formel 1,00); φ = 0,5: 1,60 / 1,65 (1,67); φ = 0,9: 7,80 / 9,34 (9,53); φ = 0,99: 20,59 / 58,55 (99,50). **Die Formel gilt, wenn T deutlich größer ist als 1/(1 − φ);** bei φ = 0,99 und T = 104 liegt sie fast um den Faktor 5 daneben. |
| Handwerte | x = (1, 2, 3, 4, 5), y = (2, 4, 5, 4, 5): r = 6/√60 = 0,7746, DW = 4,84/2,4 = 2,0167, R² = 0,6 (keine Warnung); Lag-1-Autokorrelation von x = 0,4; Differenzen von x = (1, 1, 1, 1); zirkuläres Verschieben von x = (1, 2, 3) gegen y = (1, 3, 2): p = 3/3. |

## Ehrliche Grenzen

- **Die vier Prozesse sind Beispiele.** Saison (gemeinsamer Kalender), Strukturbrüche und sehr lange Gedächtnisse sind nicht gezeigt; wie häufig die Prozesse in echten Standort-Reihen sind, zeigt die Demo nicht.
- **n_eff ist die Näherung mit den Lag-1-Autokorrelationen beider Reihen** (nach oben durch T begrenzt, mindestens 3); sie ersetzt keine vollständige Schätzung der Kovarianzstruktur (HAC-Standardfehler, Block-Bootstrap) und ist hier nicht darauf geprüft.
- **Das Verschiebungstest** vergleicht r mit allen T zyklischen Verschiebungen; für Reihen mit Trend oder Irrfahrt ist die Verteilung der Verschiebungen nicht die richtige Nullverteilung (die Zelle zeigt es), und bei T = 26 ist der kleinste p-Wert 1/26.
- **Die Warnregel R² > DW** ist ein Hinweis, kein Test; die Schwelle stammt aus der Literatur der Scheinregression und wird hier nur gemessen.
- **Echter Zusammenhang** ist hier immer eine Korrelation der Zufallsstöße (oder y = x + Rauschen); ob ein Zusammenhang kausal ist, sagt kein Verfahren.
- **Kointegration** (gemeinsame Irrfahrt mit Anpassungstempo) wird nicht getestet; die Demo zeigt nur, dass der gemeinsame Gleichlauf in den Verfahren sichtbar ist (y = x + Rauschen: 98,0 bis 100,0 %).
- **Quellen:** keine Literaturangaben; die Formeln sind durch unabhängige Rechenwege (unten) und die Messungen gestützt. Die Grenzverteilung von r bei Irrfahrten wird nur gemessen, nicht hergeleitet.

## Verifikation

- **Orakel** (`tests/test_oracle_scheinkorrelation.py`): naiver Test und Differenzen gegen `scipy.stats.pearsonr` (p-Werte auf 10 Stellen), Trendbereinigung gegen `numpy.polyfit`, n_eff-Test gegen die Definition als Schleife, Verschiebungstest gegen die Schleife über `numpy.roll`, Durbin–Watson gegen die Definition; AR(1)-Autokorrelation gegen φ^k, stationärer Start, Var(x_t) = t der Irrfahrt, und die Varianz von r gegen die geschlossene Formel auf Paaren der Länge 2 000.
- **Kern** (`tests/test_algorithm.py`, `tests/test_edge_cases.py`): Handrechnungen (Mini-Instanzen in `conftest.py`), kleinste Fälle (T = 5), konstante Reihen, Fehlerwege, Grenzen von φ, ρ und T.
- **Eigenschaften** (`tests/test_eigenschaften_scheinkorrelation.py`): jedes Gegenmittel hilft irgendwo und scheitert irgendwo (über 600 Läufe mit Schwellen, nicht an einem Seed), Invarianz gegen Verschieben und Skalieren, Symmetrie des Verschiebungstests, Trend und Niveau fallen heraus.
- **Raster und Aussagen** (`tests/test_grid.py`, `tests/test_claims.py`): Probe des Erzeugers gegen die gespeicherte, jede Zahl dieser README gegen das Raster oder die Live-Funktion, Prozentreihen mit der Rundung der App.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an seinen Grenzen), alle Presets, Adresszeile, jeder Experiment-Knopf einzeln, Layout, Footer, Laufzeit-Importe.
- **Mutationsprüfung:** `tools/mutate_demo.py` des Website-Repos mutiert den Kern `skz_algorithm` (66 Stellen, alle gesichert); 6 davon sind als gleichwertig festgehalten (ein Betrag ist unendlich oder NaN, nie null; die Zentrierung von y ändert beim Verschiebungstest nur die Skala aller r, nicht ihre Reihenfolge; ein exakter Gleichstand R² = DW). Der erste Lauf fand 7 Überlebende und führte zu Tests für p = 0 bei |r| = 1, NaN bei konstanten Reihen und das Vorzeichen des Rauschens in der gemeinsamen Irrfahrt (der Zustand steht in `tests/mutation_state.json`). Verdikte, Raster und Auswertung prüfen die übrigen Tests.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `skz_algorithm.py` | Kern: Prozesse, Paare, die fünf Verfahren, Warnregel |
| `skz_constants.py` | Regler-Grenzen, Prozesse und Verfahren, Raster der Messreihen, Farben, Presets |
| `skz_evaluation.py` | Einzelpaar, Zellen des Rasters, Live-Experimente, Verdikte |
| `skz_layout.py` | Fester Kopf (Stylesheet) |
| `skz_pdf_export.py` | PDF-Export der Hauptansicht |
| `skz_presets.py` | Permalink, Presets (Schnellstart mit Hervorhebung), Zufalls-Seed-Button |
| `skz_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `app.py` | Streamlit-App |
| `data/spurious_grid.json` | vorgerechnetes Raster (erzeugt von `tools/make_grid.py`) |
| `prs_*.py` | wörtliche Kopien aus `pearson-demo` (SplitMix64, Szenario, Pearson-Kern, Konstanten) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy für die Orakel |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Orakel, Kern, Eigenschaften, Raster, Aussagen, Verdikte, Oberfläche, Layout, Footer, Laufzeit-Importe |
| `tools/make_grid.py` | rechnet das Raster vor |

## Bewusst nicht umgesetzt

- **Kointegration, Granger-Test, Transferentropie:** das Stück zu Kointegration und gerichteter Abhängigkeit.
- **Lag-Suche:** die Kreuzkorrelationsspitze traf in der Vormessung den wahren Lag in jedem Lauf, kein Befund.
- **Saison, Strukturbrüche, lange Gedächtnisse:** nicht gezeigt.
- **HAC-Standardfehler und Block-Bootstrap:** nur als Hinweis.
- **Ursachen:** Linie *Kausale Diagramme und Adjustierung*; ein echter Zusammenhang sagt nichts über die Richtung.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Das Raster entsteht neu mit `python tools/make_grid.py` (wenige Sekunden); `tests/test_grid.py` rechnet eine feste Probe nach und schlägt an, wenn Erzeuger oder Statistik das Raster überholt haben.

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
