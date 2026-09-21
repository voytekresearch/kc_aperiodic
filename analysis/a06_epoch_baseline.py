"""Per-subject N2 baseline exponent, averaged over all 30 s N2 epochs
(2 s-equivalent Welch, nperseg = 1 s), for each of the three fit ranges. Caches:
    epochbase_<sid>.npz : B (3,), n_epochs
"""
import numpy as np
from _run import run  # noqa: F401  (also puts repo root on sys.path)
from neurodsp.spectral import compute_spectrum
from specparam import SpectralModel
from kcaperiodic import config as cfg
from kcaperiodic import core


def one(sid):
    f = cfg.CACHE_DIR / f"epochbase_{sid}.npz"
    if f.exists():
        return
    d = core.load_subject(sid, cfg.DATA_DIR); sig, fs = d["sig"], int(round(d["fs"]))
    ep = int(30 * fs); nper = int(cfg.NPERSEG_SEC * fs)
    psds, frq = [], None
    for a in range(0, len(sig) - ep, ep):
        frq, p = compute_spectrum(sig[a:a + ep], fs, nperseg=nper); psds.append(p)
    avg = np.mean(psds, 0)
    B = []
    for fr in cfg.FRANGES:
        m = SpectralModel(**cfg.SPECPARAM_KW); m.fit(frq, avg, list(fr))
        B.append(m.get_params("aperiodic", "exponent"))
    np.savez(f, B=np.array(B), n_epochs=len(psds))
    print(f"{sid}: epochbase n={len(psds)} chi={np.round(B, 2)}")


if __name__ == "__main__":
    run(one)
