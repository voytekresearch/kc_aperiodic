"""Per-subject KC-locked time-frequency power (for the ladder figure, panel C).

Morlet wavelet TFR (mne.time_frequency.tfr_array_morlet) on the stack of
KC-locked epochs. A Morlet wavelet with n_cycles = freqs/2 avoids both the
STFT windowing ringing and the DPSS-taper interference the multitaper estimator
produced against the KC's sharp edge. Epochs are cut at +/-6 s (wider than the
+/-3 s display) so wavelet edge effects fall outside the plotted window. Caches:
    tfr_<sid>.npz : freqs, times, meanLogS (freq x time), wave, twave, n
"""
import os
import numpy as np
from _run import run  # noqa: F401  (also puts repo root on sys.path)
from mne.time_frequency import tfr_array_morlet
from kcaperiodic import config as cfg
from kcaperiodic import core

FORCE = os.environ.get("KC_FORCE") == "1"   # set KC_FORCE=1 to recompute in place
HALF = 6.0              # epoch half-width (s); wider than the +/-3 s display so the
                        # wavelet cone-of-influence falls outside the plotted range
FMIN, FMAX, FSTEP = 1.0, 30.0, 0.5
DECIM = 4               # temporal decimation of the TFR (keeps files small)


def one(sid):
    f = cfg.CACHE_DIR / f"tfr_{sid}.npz"
    if f.exists() and not FORCE:
        return
    d = core.load_subject(sid, cfg.DATA_DIR); sig, fs = d["sig"], int(round(d["fs"]))
    peaks = core.align_to_negpeak(sig, fs, d["kc_onsets"])[:cfg.MAX_EVENTS]
    ep = []
    for c0 in peaks:
        c0 = int(c0); a, b = c0 - int(HALF * fs), c0 + int(HALF * fs)
        if a < 0 or b > len(sig):
            continue
        ep.append(sig[a:b])
    X = np.asarray(ep)[:, None, :]                       # (n_epochs, 1, n_times)
    freqs = np.arange(FMIN, FMAX + 1e-9, FSTEP)
    n_cycles = freqs / 2.0                               # adaptive window length
    power = tfr_array_morlet(X, sfreq=fs, freqs=freqs, n_cycles=n_cycles,
                             output="power", zero_mean=True, decim=DECIM, verbose=False)
    meanLogS = np.log10(power.mean(0)[0] + 1e-30)        # (freq, time)
    times = np.linspace(-HALF, HALF, X.shape[-1])[::DECIM][:meanLogS.shape[1]]
    wave = np.asarray(ep).mean(0)
    np.savez(f, freqs=freqs, times=times, meanLogS=meanLogS,
             wave=wave, twave=np.linspace(-HALF, HALF, len(wave)), n=len(ep))
    print(f"{sid}: tfr (Morlet)  n={len(ep)}  shape={meanLogS.shape}")


if __name__ == "__main__":
    run(one)
