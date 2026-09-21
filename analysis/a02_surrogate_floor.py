"""Per-subject spectrum-matched surrogate floor (2 s Welch, -5..+5 s).
Injects the amplitude-matched KC template into a phase-randomized surrogate of
each subject's N2 and re-runs the event-locked pipeline. Caches:
    floor_<sid>.npz    : taus, mean (3, ntaus)   full-N2 surrogate floor
    floor_ef_<sid>.npz : taus, mean, kept_frac   event-free surrogate (robustness)
"""
import numpy as np
from _run import run  # noqa: F401  (also puts repo root on sys.path)
from neurodsp.spectral import compute_spectrum
from specparam import SpectralModel
from kcaperiodic import config as cfg
from kcaperiodic import core

TAUS = np.arange(-5, 5.0001, 0.2)


def _floor_trace(sid, bg, amps, template):
    """Event-locked exponent of amplitude-matched template injections into `bg`."""
    fs = cfg.FS; half = int(cfg.LWIN * fs) // 2
    pk = int(np.argmin(template)); depth = -template.min(); L = len(template)
    spacing = int(12 * fs); start = int(7 * fs); centers = []
    for i, a in enumerate(amps):
        s = start + i * spacing
        if s + L > len(bg) - int(7 * fs):
            break
        bg[s:s + L] += template * (a / depth); centers.append(s + pk)
    return core.event_locked_welch(bg, fs, np.array(centers, int), TAUS)


def _event_free(sig, fs, kc_pk, sp_on, w=1.5):
    """Return an N2 signal with +/- w s around every KC/spindle removed (masked)."""
    ev = np.concatenate([kc_pk, (sp_on * fs).astype(int)])
    keep = np.ones(len(sig), bool); ww = int(w * fs)
    for e in ev:
        a, b = max(0, int(e) - ww), min(len(sig), int(e) + ww); keep[a:b] = False
    # synthesize a stationary signal with the event-free spectrum
    runs = []; i = 0
    while i < len(sig):
        if keep[i]:
            j = i
            while j < len(sig) and keep[j]:
                j += 1
            if j - i >= 2 * fs:
                runs.append(sig[i:j])
            i = j
        else:
            i += 1
    frq, tgt = None, []
    for r in runs:
        f, p = compute_spectrum(r, fs, nperseg=fs); frq = f; tgt.append(p)
    return frq, np.mean(tgt, 0), keep.mean()


def one(sid):
    ff, fe = cfg.CACHE_DIR / f"floor_{sid}.npz", cfg.CACHE_DIR / f"floor_ef_{sid}.npz"
    if ff.exists() and fe.exists():
        return
    tmpl = np.load(cfg.CACHE_DIR / "template.npz")["template"]
    d = core.load_subject(sid, cfg.DATA_DIR); sig, fs = d["sig"], int(round(d["fs"]))
    # real KC negative-peak amplitudes, used to amplitude-match the injections
    amps = np.abs(np.array(
        [np.median(sig[int(p) - 6 * fs:int(p) + 6 * fs]) - sig[int(p)]
         for p in core.align_to_negpeak(sig, fs, d["kc_onsets"])[:cfg.MAX_EVENTS]
         if int(p) - 6 * fs >= 0 and int(p) + 6 * fs < len(sig)]))

    # (1) full-N2 surrogate
    if not ff.exists():
        bg = core.phase_randomize(sig, cfg.SEED)
        np.savez(ff, taus=TAUS, mean=_floor_trace(sid, bg, amps, tmpl), n=len(amps))

    # (2) event-free surrogate (power-matched), robustness check
    if not fe.exists():
        kc_pk = core.align_to_negpeak(sig, fs, d["kc_onsets"])
        frq, tgt, kept = _event_free(sig, fs, kc_pk, d["spindle_onsets"])
        L = int((len(amps) * 12 + 20) * fs); ffreq = np.fft.rfftfreq(L, 1 / fs)
        mag = np.sqrt(np.interp(ffreq, frq, tgt, left=tgt[0], right=tgt[-1])); mag[0] = 0
        rng = np.random.default_rng(cfg.SEED); ph = rng.uniform(0, 2 * np.pi, len(ffreq)); ph[0] = 0
        if L % 2 == 0:
            ph[-1] = 0
        bg = np.fft.irfft(mag * np.exp(1j * ph), n=L); bg = bg / bg.std() * sig.std()
        np.savez(fe, taus=TAUS, mean=_floor_trace(sid, bg, amps, tmpl), kept_frac=kept)
    print(f"{sid}: floor / floor_ef  n_inj={len(amps)}")


if __name__ == "__main__":
    run(one)
