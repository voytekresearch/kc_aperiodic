"""Figure 2 -- the fit-range ladder.
A: KC-peak vs N2-baseline grand-mean spectra with aperiodic fits over each range.
B: event-locked exponent trace (-10..+10 s) for the three fit ranges.
C: grand-mean KC-locked spectrogram with the KC waveform overlaid.
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from specparam import SpectralModel
from _style import COLS, BAND_LAB
from kcaperiodic import config as cfg

C = str(cfg.CACHE_DIR) + "/"
FR = cfg.FRANGES


def apfit(freqs, psd, fr):
    m = SpectralModel(**cfg.SPECPARAM_KW); m.fit(freqs, psd, list(fr))
    return m.get_params("aperiodic", "offset"), m.get_params("aperiodic", "exponent")


def main():
    # A: grand-mean spectra -- peak drawn at the time of maximal exponent
    # (evpsd_pt, t = -0.5 s) so the plotted fits match the reported peak deflection
    FP = [f for f in sorted(glob.glob(C + "evpsd_*.npz")) if "evpsd_pt_" not in f]
    freqs = np.load(FP[0])["freqs"]
    mfit = (freqs >= 1) & (freqs <= 45)
    FPT = sorted(glob.glob(C + "evpsd_pt_*.npz"))
    peakG = 10 ** np.array([np.load(f)["peakP"] for f in FPT]).mean(0)   # peak-time spectrum
    baseG = 10 ** np.array([np.load(f)["baseP"] for f in FP]).mean(0)

    # B: event-locked trace, 3 bands, -10..10
    FO = sorted(glob.glob(C + "evtrace_*.npz")); taus = np.load(FO[0])["taus"]
    G = np.array([np.load(f)["mean"] for f in FO]); far = np.abs(taus) > 3
    Gd = G - np.nanmean(G[:, :, far], axis=2, keepdims=True)
    mD = np.nanmean(Gd, 0); sD = np.nanstd(Gd, 0) / np.sqrt(Gd.shape[0])
    peakDx = np.nanmax(mD, axis=1)   # per-band trace-peak deflection (label for A)

    # C: grand-mean event-locked spectrogram + KC waveform
    FT = sorted(glob.glob(C + "tfr_*.npz")); z = np.load(FT[0]); tf_f = z["freqs"]; tf_t = z["times"]
    S = np.array([np.load(f)["meanLogS"] for f in FT]).mean(0)
    S = S - S[:, np.abs(tf_t) > 2].mean(1, keepdims=True)
    wave = np.array([np.load(f)["wave"] for f in FT]).mean(0); twave = np.load(FT[0])["twave"]

    fig = plt.figure(figsize=(13.5, 8.2))
    gs = GridSpec(6, 2, width_ratios=[1, 1.5], hspace=1.1, wspace=.22,
                  left=.07, right=.965, top=.93, bottom=.08)

    for k, (fr, c, lab) in enumerate(zip(FR, COLS, BAND_LAB)):
        ax = fig.add_subplot(gs[2 * k:2 * k + 2, 0])
        ax.loglog(freqs[mfit], baseG[mfit], color="0.62", lw=1.2)
        ax.loglog(freqs[mfit], peakG[mfit], color="0.12", lw=1.7)
        m = (freqs >= fr[0]) & (freqs <= fr[1])
        ob, cb = apfit(freqs, baseG, fr); op, cp = apfit(freqs, peakG, fr)
        ax.loglog(freqs[m], 10 ** (ob - cb * np.log10(freqs[m])), color="0.55", lw=2.2, ls=(0, (4, 2)))
        ax.loglog(freqs[m], 10 ** (op - cp * np.log10(freqs[m])), color=c, lw=3.1)
        ax.axvspan(fr[0], fr[1], color=c, alpha=.06); ax.set_xlim(1, 45)
        ax.set_ylabel("power", fontsize=8.5); ax.tick_params(labelsize=7.5)
        ax.set_title(f"{lab}      Δχ = {peakDx[k]:+.2f}", color=c, fontweight="bold", fontsize=10.5, pad=3)
        if k == 0:
            ax.plot([], [], color="0.12", lw=1.7, label="KC peak spectrum")
            ax.plot([], [], color="0.62", lw=1.2, label="N2 baseline")
            ax.plot([], [], color=c, lw=3, label="aperiodic fit")
            ax.legend(frameon=False, fontsize=7, loc="lower left")
        if k == 2:
            ax.set_xlabel("frequency (Hz)", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)

    axB = fig.add_subplot(gs[0:3, 1])
    for k, (c, lab) in enumerate(zip(COLS, BAND_LAB)):
        axB.plot(taus, mD[k], color=c, lw=2.6, label=lab)
        axB.fill_between(taus, mD[k] - sD[k], mD[k] + sD[k], color=c, alpha=.16)
    axB.axvline(0, ls="--", color="0.45", lw=1.1); axB.axhline(0, ls=":", color="0.6", lw=.8)
    axB.set_xlim(-10, 10); axB.set_xlabel("time relative to K-complex (s)", fontsize=10)
    axB.set_ylabel("Δ aperiodic exponent\n(vs. N2 baseline)", fontsize=10)
    axB.legend(frameon=False, fontsize=9, loc="upper right")
    axB.spines[["top", "right"]].set_visible(False)

    axC = fig.add_subplot(gs[3:6, 1])
    pcm = axC.pcolormesh(tf_t, tf_f, S, cmap="magma", shading="gouraud")
    cb = plt.colorbar(pcm, ax=axC, fraction=.046, pad=.02)
    cb.set_label("Δ log₁₀ power", fontsize=9); cb.ax.tick_params(labelsize=7.5)
    wv = wave - wave.mean(); wv = 7 + (wv / np.max(np.abs(wv))) * 7
    axC.plot(twave, wv, color="w", lw=1.8, path_effects=[pe.withStroke(linewidth=3, foreground="k")])
    axC.set_xlim(-3, 3); axC.set_ylim(1, 30)
    axC.set_xlabel("time relative to K-complex (s)", fontsize=10)
    axC.set_ylabel("frequency (Hz)", fontsize=10); axC.tick_params(labelsize=8)

    fig.text(.02, .955, "A", fontweight="bold", fontsize=16)
    fig.text(.40, .955, "B", fontweight="bold", fontsize=16)
    fig.text(.40, .475, "C", fontweight="bold", fontsize=16)
    out = cfg.FIG_DIR / "fig2_ladder.png"
    plt.savefig(out, dpi=190, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    main()
