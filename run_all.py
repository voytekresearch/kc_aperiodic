"""End-to-end driver: build the template, generate every per-subject cache,
render all figures, and print the statistics. Run from the repo root:

    python run_all.py            # all subjects in config.SUBJECTS
    python run_all.py 01-02-0019 # (re)generate caches for one subject only

Caches already present in data/ are skipped, so re-running is cheap.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ANALYSIS = ["a00_template", "a01_event_locked", "a02_surrogate_floor",
            "a03_amplitude", "a04_regression", "a05_spectrogram", "a06_epoch_baseline"]
FIGURES = ["fig2_ladder", "fig3_amplitude", "fig4_surrogate_floor",
           "fig5_regression", "figS1_alignment"]


def _run_script(folder, name, argv):
    path = ROOT / folder / f"{name}.py"
    sys.argv = [str(path)] + argv
    sys.path.insert(0, str(ROOT / folder))
    print(f"\n----- {folder}/{name}.py -----")
    runpy.run_path(str(path), run_name="__main__")
    sys.path.pop(0)


def main():
    subj_args = sys.argv[1:]
    # a00 builds the grand-average template across ALL subjects; never subset it
    _run_script("analysis", "a00_template", [])
    for name in ANALYSIS[1:]:
        _run_script("analysis", name, subj_args)
    for name in FIGURES:
        _run_script("figures", name, [])
    _run_script("stats", "stats", [])


if __name__ == "__main__":
    main()
