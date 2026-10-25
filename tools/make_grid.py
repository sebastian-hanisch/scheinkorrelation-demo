"""Rechnet das Raster der Messreihen vor (E1 bis E4) und schreibt data/spurious_grid.json.

    python tools/make_grid.py            # volles Raster, mehrere Prozesse (einige Minuten)
    python tools/make_grid.py --quick    # kleines Probe-Raster für Tests (nicht in data/spurious_grid.json)

Das Raster ist deterministisch (SplitMix64, feste Seeds); tests/test_grid.py rechnet eine feste Probe nach und vergleicht sie mit der Datei."""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import skz_constants as C  # noqa: E402
import skz_evaluation as ev  # noqa: E402

JOBS = {"fa": ev.fa_cell, "rel": ev.rel_cell}


def _run(task):
    kind, args = task
    return kind, JOBS[kind](*args)


def main() -> None:
    quick = "--quick" in sys.argv
    tasks = [("fa", (key, T)) for T in sorted(C.GRID_TS, reverse=True) for key in C.GRID_PROCESSES]                 # die teuren Zellen zuerst
    tasks += [("rel", (sc, rho, T)) for T in sorted(C.REL_TS, reverse=True) for sc in C.REL_SCENARIOS for rho in ((0.0,) if sc == "coint" else C.REL_RHOS)]
    if quick:
        tasks = [("fa", ("walk", 26)), ("rel", ("walk", 0.3, 52))]
    t0 = time.time()
    out = {k: [] for k in JOBS}
    with ProcessPoolExecutor(max_workers=10) as pool:
        for kind, row in pool.map(_run, tasks, chunksize=1):
            out[kind].append(row)
    grid = {"meta": {"seeds": list(C.GRID_SEEDS), "seconds": round(time.time() - t0)}, **out, "probe": ev.grid_probe()}
    dest = ROOT / "data" / ("spurious_grid_quick.json" if quick else "spurious_grid.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{dest} geschrieben: " + ", ".join(f"{len(v)} {k}" for k, v in out.items()) + f", {grid['meta']['seconds']} s")


if __name__ == "__main__":
    main()
