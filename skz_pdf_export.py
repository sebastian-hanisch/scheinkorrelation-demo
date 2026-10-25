"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Ergebnis und die fünf Verfahren der Hauptansicht. Kein Gedankenstrich und kein Euro-Zeichen (Kernschriften)."""

from fpdf import FPDF

import skz_constants as C
import skz_evaluation as ev


def _text(value: str) -> str:
    """Zeichen, die die Kernschriften nicht kennen, ersetzen (Gedankenstrich, Pfeile, Fettdruck-Markierung, griechische und mathematische Zeichen)."""
    table = {"–": "-", "—": "-", "→": "->", "**": "", "€": "EUR", "·": "*", "≤": "<=", "≥": ">=", "−": "-", "×": "x", "„": '"', "“": '"', "ρ": "rho", "φ": "phi", "σ": "sigma", "τ": "tau",
             "²": "^2", "√": "sqrt ", "…": "...", "π": "pi"}
    for old, new in table.items():
        value = value.replace(old, new)
    return value.encode("latin-1", "replace").decode("latin-1")


def _para(pdf: FPDF, height: float, text: str) -> None:
    """Absatz über die volle Breite; danach steht der Cursor wieder am linken Rand."""
    pdf.multi_cell(0, height, _text(text), new_x="LMARGIN", new_y="NEXT")


def build_pdf(a: ev.Analysis) -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _text("Scheinkorrelation: zwei Reihen ohne Zusammenhang"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    extra = f"   phi: {s.phi:g}" if s.process == "ar1" else (f"   Trend je Woche: {s.slope:g}" if s.process == "trend" else "")
    _para(pdf, 5.5, f"Prozess: {C.PROCESSES[s.process]}{extra}   T: {s.T} Wochen   Zusammenhang rho: {s.rho:g}   Zufalls-Seed: {s.seed}")
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Ergebnis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    _para(pdf, 5.5, ev.verdict_main(a)[1])
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Verfahren (5 %-Niveau)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for key, name in C.METHOD_LONG.items():
        r, p = a.results[key]
        _para(pdf, 5.5, f"{name}: r = {r:.3f}, p = {p:.4f}, {'verwirft Unabhängigkeit' if key in a.found else 'verwirft nicht'}")
    _para(pdf, 5.5, f"Warnregel R^2 > DW: R^2 = {a.r2:.3f}, DW = {a.dw:.3f}, {'Warnung' if a.warning else 'keine Warnung'}")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    _para(pdf, 4.5, "Simulierte Reihen mit bekannter Wahrheit. Sebastian Hanisch - Operations Research und Machine Learning.")
    return bytes(pdf.output())
