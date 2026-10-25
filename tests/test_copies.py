"""Die kopierten Bausteine (wörtlich, Quelle im Kopf der Datei): Vergleich mit den Originalen, solange sie im Portfolio-Ordner liegen."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {name: ("pearson-demo", name) for name in ("prs_rng.py", "prs_constants.py", "prs_scenario.py", "prs_algorithm.py")}


@pytest.mark.parametrize("copy", list(SOURCES))
def test_copy_is_verbatim_apart_from_the_header(copy):
    folder, original = SOURCES[copy]
    source = ROOT.parent / folder / original
    if not source.exists():
        pytest.skip("Original liegt nicht im Portfolio-Ordner")
    mine = (ROOT / copy).read_text(encoding="utf-8").splitlines()
    assert mine[0].startswith(f"# Wörtliche Kopie von {folder}/{original}")
    assert mine[1:] == source.read_text(encoding="utf-8").splitlines(), "die Kopie weicht vom Original ab: Änderungen gehören ins Original, danach neu kopieren"


def test_the_copies_run_on_their_own():
    from prs_rng import SplitMix64
    from prs_scenario import VRng
    assert [SplitMix64(0).next() for _ in range(1)] == [16294208416658607535]
    assert [int(v) for v in VRng(0).u64(3)] == [16294208416658607535, 7960286522194355700, 487617019471545679]
