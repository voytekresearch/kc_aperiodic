"""Per-subject waveform-regression control (Gyurkovics et al., 2022; 2 s Welch).
Regresses the KC template out of a +/-1 s window at the negative peak and
re-estimates the event-locked exponent on the residual signal. Caches:
    regtr_<sid>.npz : taus, Eo (3,ntaus), Er (3,ntaus), wave, resid, twave, reg_half
                      (Eo = original, Er = after regression; wave/resid = +/-4 s voltage)
"""
import numpy as np
from _run import run  # noqa: F401  (also puts repo root on sys.path)
from kcaperiodic import config as cfg
from kcaperiodic import core

TAUS = np.arange(-4, 4.0001, 0.25)


def one(sid):
    f = cfg.CACHE_DIR / f"regtr_{sid}.npz"
    if f.exists():
        return
    fs = cfg.FS; half = int(cfg.LWIN * fs) // 2; LH = int(4 * fs)
    tmpl = np.load(cfg.CACHE_DIR / "template.npz")["template"]
    pk = int(np.argmin(tmpl)); tc = tmpl[pk - half:pk + half].copy(); tc = tc - tc.mean()
    d = core.load_subject(sid, cfg.DATA_DIR); sig = d["sig"]
    peaks = core.align_to_negpeak(sig, fs, d["kc_onsets"])[:cfg.MAX_EVENTS]
    nb = len(cfg.FRANGES)
    Eo = np.full((len(peaks), nb, len(TAUS)), np.nan); Er = np.full_like(Eo, np.nan)
    W, R = [], []
    for i, c0 in enumerate(peaks):
        c0 = int(c0)
        if c0 - LH - half < 0 or c0 + LH + half > len(sig):
            continue
        seg = sig.copy()
        seg[c0 - half:c0 + half] = core.regress_out_template(sig[c0 - half:c0 + half], tc)
        W.append(sig[c0 - LH:c0 + LH] - sig[c0 - LH:c0 - LH + int(0.3 * fs)].mean())
        R.append(seg[c0 - LH:c0 + LH] - seg[c0 - LH:c0 - LH + int(0.3 * fs)].mean())
        for j, ta in enumerate(TAUS):
            c = c0 + int(round(ta * fs)); a, b = c - half, c + half
            if a < 0 or b > len(sig):
                continue
            Eo[i, :, j] = core.welch_exponents(sig[a:b], fs)
            Er[i, :, j] = core.welch_exponents(seg[a:b], fs)
    np.savez(f, taus=TAUS, Eo=np.nanmean(Eo, 0), Er=np.nanmean(Er, 0),
             wave=np.mean(W, 0), resid=np.mean(R, 0),
             twave=np.arange(-LH, LH) / fs, reg_half=half / fs)
    print(f"{sid}: regtr  n={len(W)}")


if __name__ == "__main__":
    run(one)
