"""Figure 5 -- time-resolved waveform regression (Gyurkovics et al., 2022).
A: grand-mean voltage before/after regressing the KC template out of the
   +/-1 s window (highlighted); shown over a longer +/-4 s window.
B-D: event-locked exponent before vs after regression, one panel per fit range.
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
from _style import PINK
from kcaperiodic import config as cfg

C = str(cfg.CACHE_DIR) + "/"
BLACK = "#111111"


def main():
    F = sorted(glob.glob(C + "regtr_*.npz"))
    z0 = np.load(F[0]); taus = z0["taus"]; tw = z0["twave"]; reg = float(z0["reg_half"])
    EoS = np.array([np.load(f)["Eo"] for f in F])   # (nsub, 3, ntaus)
    ErS = np.array([np.load(f)["Er"] for f in F])
    wave = np.mean([np.load(f)["wave"] for f in F], 0)
    resid = np.mean([np.load(f)["resid"] for f in F], 0)

    base = np.abs(taus) > 3
    # baseline-subtract each subject, then grand mean + bootstrap 95% CI across subjects
    Eob = EoS - np.nanmean(EoS[:, :, base], axis=2, keepdims=True)
    Erb = ErS - np.nanmean(ErS[:, :, base], axis=2, keepdims=True)
    n = Eob.shape[0]; rng0 = np.random.default_rng(cfg.SEED)
    bo = np.empty((cfg.N_BOOT, 3, len(taus))); be = np.empty_like(bo)
    for i in range(cfg.N_BOOT):
        bo[i] = np.nanmean(Eob[rng0.integers(0, n, n)], 0)
        be[i] = np.nanmean(Erb[rng0.integers(0, n, n)], 0)
    mEo, mEr = np.nanmean(Eob, 0), np.nanmean(Erb, 0)
    loEo, hiEo = np.nanpercentile(bo, [2.5, 97.5], axis=0)
    loEr, hiEr = np.nanpercentile(be, [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(4, 1, figsize=(7.5, 10.6), sharex=True,
                           gridspec_kw={'height_ratios': [1.1, 1, 1, 1], 'hspace': .16})
    XL = (-4, 4)

    ax[0].axvspan(-reg, reg, color=PINK, alpha=.10)
    ax[0].plot(tw, wave, color=BLACK, lw=1.8, label="original")
    ax[0].plot(tw, resid, color=PINK, lw=1.8, label="waveform regressed out")
    ax[0].axhline(0, color="0.85", lw=.7)
    ax[0].set_ylabel("µV"); ax[0].legend(frameon=False, fontsize=9, loc="lower right")
    ax[0].set_title("A", loc="left", fontweight="bold", fontsize=14)
    ax[0].text(0, ax[0].get_ylim()[1] * 0.9, "regressed window", ha="center", fontsize=8, color="#a61e4d")
    ax[0].spines[["top", "right"]].set_visible(False)

    titles = ["B      broadband (1–45 Hz)", "C      high-band (10–45 Hz)", "D      high-band (20–45 Hz)"]
    for k in range(3):
        a = ax[k + 1]
        a.plot(taus, mEo[k], color=BLACK, lw=2.2, label="original")
        a.fill_between(taus, loEo[k], hiEo[k], color=BLACK, alpha=.15, lw=0)
        a.plot(taus, mEr[k], color=PINK, lw=2.2, label="after regression")
        a.fill_between(taus, loEr[k], hiEr[k], color=PINK, alpha=.18, lw=0)
        a.axvline(0, color="0.7", lw=.8, ls=":"); a.axhline(0, color="0.85", lw=.7, ls=":")
        a.set_ylabel("Δ exponent"); a.legend(frameon=False, fontsize=9, loc="upper right")
        a.set_title(titles[k], loc="left", fontweight="bold", fontsize=11)
        a.spines[["top", "right"]].set_visible(False)
    ax[3].set_xlabel("time relative to K-complex (s)"); ax[3].set_xlim(*XL)

    plt.tight_layout()
    out = cfg.FIG_DIR / "fig5_regression.png"
    plt.savefig(out, dpi=190, bbox_inches="tight"); print("saved", out, "n=%d" % len(F))


if __name__ == "__main__":
    main()
