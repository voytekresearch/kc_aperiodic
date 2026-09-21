"""
Core analysis functions for
"Aperiodic dynamics track cortical state shifts during sleep K-complexes."

Two spectral estimators are used and kept explicit:
  * Welch, 2 s windows (`welch_exponents`, `event_locked_welch`) -- the primary
    time-resolved estimator, matched to the K-complex's low-frequency footprint.
  * multitaper, 1 s windows (`mt_exponents`, `amplitude_deflection`) -- used for
    the per-event amplitude and regression controls.

Every analysis (real KCs, surrogate floor, waveform regression) uses the same
functions so the numbers are directly comparable.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from neurodsp.spectral import compute_spectrum
from specparam import SpectralModel

from .config import (FS, FRANGES, SPECPARAM_KW, LWIN, NPERSEG_SEC, WIN_MT,
                     MT_BANDWIDTH, BASELINE_S)

mne.set_log_level("ERROR")


# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #
def _kc_spindle_onsets(desc, onset):
    grp = (desc.str.extract(r'[gG]roup[nN]ame="([^"]+)"')[0].fillna(desc).str.lower())
    kc = onset[grp.str.contains("kcomplex").to_numpy()]
    sp = onset[grp.str.contains("spindle").to_numpy()]
    return kc.astype(float), sp.astype(float)


def load_subject(subject_id, data_root):
    """Load the cleaned C3 signal (uV) and expert KC/spindle onsets (s)."""
    d = Path(data_root) / subject_id / "cleaned"
    raw = mne.io.read_raw_fif(d / f"{subject_id}_cleaned_raw.fif", preload=False)
    c3 = next(c for c in raw.ch_names if "C3" in c.upper())
    sig = raw.get_data(picks=[c3], units="uV")[0].astype(float)
    fs = int(round(raw.info["sfreq"]))
    ann = pd.read_csv(d / f"{subject_id}_annotations.csv")
    kc, sp = _kc_spindle_onsets(ann["description"].astype(str),
                                ann["onset_s"].astype(float).to_numpy())
    return dict(sig=sig, fs=fs, kc_onsets=kc, spindle_onsets=sp)


def align_to_negpeak(sig, fs, onsets, search_s=1.0):
    """Re-align each expert onset to the KC negative peak within `search_s`."""
    peaks = []
    for on in onsets:
        c = int(round(on * fs))
        a, b = c, min(c + int(search_s * fs), len(sig))
        if b - a > 10:
            peaks.append(a + int(np.argmin(sig[a:b])))
    return np.array(peaks, int)


# --------------------------------------------------------------------------- #
#  Spectral parameterization of one window
# --------------------------------------------------------------------------- #
def welch_exponents(x, fs=FS, franges=FRANGES, nperseg_sec=NPERSEG_SEC):
    """Aperiodic exponent of one window, Welch PSD re-fit over each fit range."""
    f, p = compute_spectrum(x, fs, nperseg=int(nperseg_sec * fs))
    out = []
    for fr in franges:
        fm = SpectralModel(**SPECPARAM_KW)
        try:
            fm.fit(f, p, list(fr)); out.append(fm.get_params("aperiodic", "exponent"))
        except Exception:
            out.append(np.nan)
    return np.array(out)


def mt_exponents(x, fs=FS, franges=FRANGES, return_r2=False):
    """Aperiodic exponent of one window, multitaper PSD re-fit over each range."""
    psd, freqs = mne.time_frequency.psd_array_multitaper(
        x[None, :], sfreq=fs, fmin=min(f[0] for f in franges),
        fmax=max(f[1] for f in franges), bandwidth=MT_BANDWIDTH, verbose=False)
    psd = np.clip(psd[0], 1e-30, None)
    exps, r2 = [], np.nan
    for i, fr in enumerate(franges):
        fm = SpectralModel(**SPECPARAM_KW)
        try:
            fm.fit(freqs, psd, list(fr)); e = fm.get_params("aperiodic", "exponent")
            if i == 0:
                r2 = fm.get_metrics("gof", "rsquared")
        except Exception:
            e = np.nan
        exps.append(e)
    return (np.array(exps), r2) if return_r2 else np.array(exps)


# --------------------------------------------------------------------------- #
#  Event-locked exponent trace (2 s Welch)
# --------------------------------------------------------------------------- #
def event_locked_welch(sig, fs, peaks, taus, franges=FRANGES, win=LWIN):
    """Event-locked exponent (mean over events) at each tau, per fit range."""
    half = int(win * fs) // 2
    E = np.full((len(peaks), len(franges), len(taus)), np.nan)
    for i, c0 in enumerate(peaks):
        c0 = int(c0)
        for j, ta in enumerate(taus):
            c = c0 + int(round(ta * fs)); a, b = c - half, c + half
            if a < 0 or b > len(sig):
                continue
            E[i, :, j] = welch_exponents(sig[a:b], fs, franges)
    return np.nanmean(E, 0)


# --------------------------------------------------------------------------- #
#  Per-event amplitude + deflection (1 s multitaper)
# --------------------------------------------------------------------------- #
def amplitude_deflection(sig, fs, peaks, franges=FRANGES,
                         epoch_s=6.0, baseline_s=BASELINE_S, win=WIN_MT):
    """Per-event KC amplitude and exponent deflection (peak minus +/- baseline)."""
    half = int(win * fs) // 2; ep = int(epoch_s * fs)
    amps, defls = [], [[] for _ in franges]
    for pk in peaks:
        pk = int(pk)
        if pk - ep < 0 or pk + ep > len(sig):
            continue
        amp = np.median(sig[pk - ep:pk + ep]) - sig[pk]
        e_pk = mt_exponents(sig[pk - half:pk + half], fs, franges)
        e_b1 = mt_exponents(sig[pk - int(baseline_s * fs) - half:
                                pk - int(baseline_s * fs) + half], fs, franges)
        e_b2 = mt_exponents(sig[pk + int(baseline_s * fs) - half:
                                pk + int(baseline_s * fs) + half], fs, franges)
        row = [e_pk[k] - np.nanmean([e_b1[k], e_b2[k]]) for k in range(len(franges))]
        if np.any(np.isnan(row)) or np.isnan(amp):
            continue
        amps.append(amp)
        for k in range(len(franges)):
            defls[k].append(row[k])
    return np.array(amps), [np.array(d) for d in defls]


# --------------------------------------------------------------------------- #
#  KC template + phase-randomized surrogate
# --------------------------------------------------------------------------- #
def build_kc_template(sig, fs, kc_onsets, spindle_onsets, pre=1.0, post=1.5, iso_s=10.0):
    """Grand-average waveform of isolated KCs (neg peak at sample int(pre*fs))."""
    events = np.sort(np.concatenate([kc_onsets, spindle_onsets]))
    n = int((pre + post) * fs); segs = []
    for on in kc_onsets:
        others = events[events != on]
        if others.size and np.min(np.abs(others - on)) <= iso_s:
            continue
        c = int(on * fs); a, b = c, min(c + int(fs), len(sig))
        if b - a < 10:
            continue
        pk = a + int(np.argmin(sig[a:b])); s = pk - int(pre * fs); e = s + n
        if s < 0 or e > len(sig):
            continue
        segs.append(sig[s:e] - sig[s:s + int(pre * fs)].mean())
    return np.mean(segs, axis=0)


def phase_randomize(x, seed=0):
    """Fourier surrogate: preserve the power spectrum, randomize phases.

    Keeps the exact power spectrum (Theiler et al., 1992; Schreiber & Schmitz,
    2000) while destroying all time-locked structure, so the aperiodic exponent
    is stationary by construction and no events remain.
    """
    rng = np.random.default_rng(seed)
    X = np.fft.rfft(x); ph = rng.uniform(0, 2 * np.pi, len(X)); ph[0] = 0
    if len(x) % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(x))


def surrogate_floor_deflections(sig, fs, template, amps, taus=None, franges=FRANGES,
                                win=LWIN, seed=0, spacing_s=12.0, start_s=7.0):
    """Waveform-only floor: inject the amplitude-matched template into a
    phase-randomized surrogate of `sig` and measure the event-locked exponent.

    Returns the mean floor trace (per fit range) if `taus` is given, else the
    per-injection deflection at the injection center (peak minus +/- baseline).
    """
    half = int(win * fs) // 2
    pk = int(np.argmin(template)); depth = -template.min(); L = len(template)
    bg = phase_randomize(sig, seed)
    spacing = int(spacing_s * fs); start = int(start_s * fs); centers = []
    for i, a in enumerate(amps):
        s = start + i * spacing
        if s + L > len(bg) - int(start_s * fs):
            break
        bg[s:s + L] += template * (a / depth); centers.append(s + pk)
    if taus is not None:                                    # mean floor trace
        return event_locked_welch(bg, fs, np.array(centers, int), taus, franges, win)
    _, defls = amplitude_deflection(bg, fs, np.array(centers, int), franges)  # fallback
    return defls


# --------------------------------------------------------------------------- #
#  Waveform regression (Gyurkovics et al., 2022)
# --------------------------------------------------------------------------- #
def regress_out_template(window, template_central):
    """Least-squares removal of the (mean-subtracted) template from a window."""
    tc = template_central - template_central.mean()
    b = np.dot(window - window.mean(), tc) / np.dot(tc, tc)
    return window - b * tc
