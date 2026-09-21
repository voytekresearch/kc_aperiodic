"""Per-subject event-locked exponent trace (2 s Welch, -10..+10 s, 3 fit ranges)
and the KC-peak vs N2-baseline spectra. Caches:
    evtrace_<sid>.npz : taus, mean (3, ntaus)   real KC-locked exponent
    evpsd_<sid>.npz   : freqs, peakP, baseP     grand-mean log10 PSDs
"""
import numpy as np
from _run import run  # noqa: F401  (also puts repo root on sys.path)
from neurodsp.spectral import compute_spectrum
from kcaperiodic import config as cfg
from kcaperiodic import core

# dense sampling near the peak, coarse in the flat tails
TAUS = np.concatenate([np.arange(-10, -5, 0.5), np.arange(-5, 5.0001, 0.25),
                       np.arange(5.5, 10.0001, 0.5)])


def one(sid):
    fe, fp = cfg.CACHE_DIR / f"evtrace_{sid}.npz", cfg.CACHE_DIR / f"evpsd_{sid}.npz"
    if fe.exists() and fp.exists():
        return
    d = core.load_subject(sid, cfg.DATA_DIR); sig, fs = d["sig"], d["fs"]
    peaks = core.align_to_negpeak(sig, fs, d["kc_onsets"])[:cfg.MAX_EVENTS]
    mean = core.event_locked_welch(sig, fs, peaks, TAUS)
    np.savez(fe, taus=TAUS, mean=mean, n=len(peaks))

    # KC-peak and N2-baseline spectra (2 s Welch)
    half = int(cfg.LWIN * fs) // 2; bs = int(cfg.BASELINE_S * fs); nper = int(cfg.NPERSEG_SEC * fs)
    pk, bl, frq = [], [], None
    for c in peaks:
        c = int(c)
        if c - bs - half < 0 or c + bs + half > len(sig):
            continue
        frq, pp = compute_spectrum(sig[c - half:c + half], fs, nperseg=nper)
        _, b1 = compute_spectrum(sig[c - bs - half:c - bs + half], fs, nperseg=nper)
        _, b2 = compute_spectrum(sig[c + bs - half:c + bs + half], fs, nperseg=nper)
        pk.append(np.log10(np.clip(pp, 1e-30, None)))
        bl.append(np.log10(np.clip((b1 + b2) / 2, 1e-30, None)))
    np.savez(fp, freqs=frq, peakP=np.mean(pk, 0), baseP=np.mean(bl, 0))
    print(f"{sid}: evtrace/evpsd  n={len(peaks)}")


if __name__ == "__main__":
    run(one)
