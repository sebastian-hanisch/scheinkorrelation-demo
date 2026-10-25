"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def five():
    """x = (1, 2, 3, 4, 5), y = (2, 4, 5, 4, 5). Mittel 3 und 4; Σ(x−x̄)(y−ȳ) = 4 + 0 + 0 + 0 + 2 = 6, Σ(x−x̄)² = 10, Σ(y−ȳ)² = 6, also r = 6/√60 = 0,7746, b = 0,6.
    Residuen u = (y − 4) − 0,6 (x − 3) = (−0,8; 0,6; 1; −0,6; −0,2), Σu² = 2,4, Σ(Δu)² = 1,96 + 0,16 + 2,56 + 0,16 = 4,84, DW = 4,84/2,4 = 2,0167, R² = 0,6 (kleiner als DW: keine Warnung).
    Lag-1-Autokorrelation von x: Σ(x_t − 3)(x_{t+1} − 3) = 2 + 0 + 0 + 2 = 4, durch Σ(x−3)² = 10: 0,4. Differenzen von x: (1, 1, 1, 1)."""
    return [1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 5.0, 4.0, 5.0]


@pytest.fixture
def three():
    """x = (1, 2, 3), y = (1, 3, 2): zentriert (−1, 0, 1) und (−1, 1, 0). Verschiebung 0: Σ = 1, r = 1/2; Verschiebung 1 (y → (2, 1, 3), zentriert (0, −1, 1)): Σ = 1, r = 1/2;
    Verschiebung 2 (y → (3, 2, 1), zentriert (1, 0, −1)): Σ = −2, r = −1. Alle |r| ≥ 1/2, also p = 3/3 = 1."""
    return [1.0, 2.0, 3.0], [1.0, 3.0, 2.0]
