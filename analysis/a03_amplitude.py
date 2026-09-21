"""Per-subject amplitude-tracking control (1 s multitaper). Caches:
    ampdefl_<sid>.npz : amp, defl (Nev, 3)           per-event amplitude & deflection
    rfmin_<sid>.npz   : fmins, amp, defl (Nev, nFmin) deflection vs lower fit bound
"""
import numpy as np
from _run import run  # noqa: F401  (also puts repo root on sys.path)
from kcaperiodic import config as cfg
from kcaperiodic import core

FMINS = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20]


def one(sid):
    fa, fr = cfg.CACHE_DIR / f"ampdefl_{sid}.npz", cfg.CACHE_DIR / f"rfmin_{sid}.npz"
    if fa.exists() and fr.exists():
        return
    d = core.load_subject(sid, cfg.DATA_DIR); sig, fs = d["sig"], d["fs"]
    peaks = core.align_to_negpeak(sig, fs, d["kc_onsets"])[:cfg.MAX_EVENTS]

    if not fa.exists():
        amps, defls = core.amplitude_deflection(sig, fs, peaks, cfg.FRANGES)
        np.savez(fa, amp=amps, defl=np.array(defls).T)

    if not fr.exists():
        FR = [(float(fm), 45.0) for fm in FMINS]
        amps, defls = core.amplitude_deflection(sig, fs, peaks, FR)
        np.savez(fr, fmins=np.array(FMINS), amp=amps, defl=np.array(defls).T)
    print(f"{sid}: ampdefl / rfmin")


if __name__ == "__main__":
    run(one)
