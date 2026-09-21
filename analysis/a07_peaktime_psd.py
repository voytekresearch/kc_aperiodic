"""Per-subject grand-mean spectrum at the time of peak aperiodic exponent
(the event-locked trace peaks ~0.5 s before the KC negative peak). Used to draw
Figure 2A so the plotted spectra correspond to the reported peak deflection
rather than the value at the negative peak (t = 0). Caches:
    evpsd_pt_<sid>.npz : freqs, peakP  (log10 grand-mean PSD at t = PEAK_TAU)
"""
import os
import numpy as np
from _run import run  # noqa: F401  (also puts repo root on sys.path)
from neurodsp.spectral import compute_spectrum
from kcaperiodic import config as cfg
from kcaperiodic import core

FORCE = os.environ.get("KC_FORCE") == "1"
PEAK_TAU = -0.5   # s; time of the grand-mean broadband exponent trace peak


def one(sid):
    f = cfg.CACHE_DIR / f"evpsd_pt_{sid}.npz"
    if f.exists() and not FORCE:
        return
    d = core.load_subject(sid, cfg.DATA_DIR); sig, fs = d["sig"], int(round(d["fs"]))
    peaks = core.align_to_negpeak(sig, fs, d["kc_onsets"])[:cfg.MAX_EVENTS]
    half = int(cfg.LWIN * fs) // 2; nper = int(cfg.NPERSEG_SEC * fs)
    off = int(round(PEAK_TAU * fs))
    sp, frq = [], None
    for c0 in peaks:
        c = int(c0) + off; a, b = c - half, c + half
        if a < 0 or b > len(sig):
            continue
        frq, p = compute_spectrum(sig[a:b], fs, nperseg=nper)
        sp.append(np.log10(np.clip(p, 1e-30, None)))
    np.savez(f, freqs=frq, peakP=np.mean(sp, 0))
    print(f"{sid}: evpsd_pt (t={PEAK_TAU}s)  n={len(sp)}")


if __name__ == "__main__":
    run(one)
