"""Figure 4 -- the spectrum-matched surrogate floor.
A-C: construction. Phase-randomized surrogate preserves the N2 power spectrum
     (incl. the spindle bump) while destroying time-locked events; the KC
     template is injected amplitude-matched.
D-F: real KC-locked exponent vs the surrogate floor, per fit range, with the
     bootstrapped residual (real peak minus floor peak).
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from neurodsp.spectral import compute_spectrum
from _style import COLS, GREY, PINK
from kcaperiodic import config as cfg
from kcaperiodic import core

C = str(cfg.CACHE_DIR) + "/"
FS = cfg.FS


def main():
    # setup panels from one subject
    sid = cfg.SUBJECTS[0]; d = core.load_subject(sid, cfg.DATA_DIR); sig = d["sig"]
    sur = core.phase_randomize(sig, cfg.SEED)
    pk = core.align_to_negpeak(sig, FS, d["kc_onsets"]); c = int(pk[len(pk) // 2])
    seg = slice(c - int(6 * FS), c + int(6 * FS)); tt = np.arange(int(12 * FS)) / FS - 6
    frq, pr = compute_spectrum(sig, FS, nperseg=FS * 4)
    _, ps = compute_spectrum(sur, FS, nperseg=FS * 4); mf = (frq >= 1) & (frq <= 45)
    TMPL = np.load(C + "template.npz")["template"]; tw = np.arange(len(TMPL)) / FS - 1.0

    # result panels
    S = [f.split("floor_")[1][:10] for f in sorted(glob.glob(C + "floor_*.npz"))
         if "floor_ef_" not in f]
    taus = np.load(C + f"evtrace_{S[0]}.npz")["taus"]; tflo = np.load(C + f"floor_{S[0]}.npz")["taus"]
    REAL = np.array([np.load(C + f"evtrace_{s}.npz")["mean"] for s in S])
    FLOOR = np.array([np.load(C + f"floor_{s}.npz")["mean"] for s in S])
    br, bf = np.abs(taus) > 3, np.abs(tflo) > 3
    # baseline-subtract each subject, then mean + bootstrap 95% CI across subjects
    Rb = REAL - np.nanmean(REAL[:, :, br], axis=2, keepdims=True)
    Fb = FLOOR - np.nanmean(FLOOR[:, :, bf], axis=2, keepdims=True)
    mR = np.nanmean(Rb, 0); mF = np.nanmean(Fb, 0)
    n = Rb.shape[0]; rng0 = np.random.default_rng(cfg.SEED)
    bR = np.empty((cfg.N_BOOT,) + mR.shape); bF = np.empty((cfg.N_BOOT,) + mF.shape)
    for i in range(cfg.N_BOOT):
        bR[i] = np.nanmean(Rb[rng0.integers(0, n, n)], 0)
        bF[i] = np.nanmean(Fb[rng0.integers(0, n, n)], 0)
    loR, hiR = np.nanpercentile(bR, [2.5, 97.5], axis=0)
    loF, hiF = np.nanpercentile(bF, [2.5, 97.5], axis=0)
    rng = np.random.default_rng(cfg.SEED); resid = np.zeros((cfg.N_BOOT, 3))

    def _pk(m, base):
        return np.array([np.nanmax(m[k]) - np.nanmean(m[k][base]) for k in range(3)])

    for i in range(cfg.N_BOOT):
        idx = rng.integers(0, len(S), len(S))
        resid[i] = _pk(np.nanmean(REAL[idx], 0), br) - _pk(np.nanmean(FLOOR[idx], 0), bf)

    fig = plt.figure(figsize=(14, 8.6))
    gs = GridSpec(2, 3, hspace=.42, wspace=.28, left=.06, right=.97, top=.9, bottom=.08)

    axA = fig.add_subplot(gs[0, 0])
    axA.plot(tt, sig[seg], color="0.12", lw=.8)
    axA.plot(tt, sur[seg] - 300, color=PINK, lw=.8)
    axA.axvline(0, color="0.7", lw=.7, ls=":"); axA.set_yticks([])
    axA.set_xlabel("time re: KC (s)", fontsize=11); axA.tick_params(labelsize=10)
    axA.set_ylim(-430, 150)
    axA.text(-5.9, 145, "real N2 (KC at 0)", color="0.12", fontsize=11.5, va="top")
    axA.text(-5.9, -95, "surrogate", color=PINK, fontsize=11.5, va="top")
    axA.spines[["top", "right", "left"]].set_visible(False)
    axA.text(-0.02, 1.04, "A", transform=axA.transAxes, fontweight="bold", fontsize=16)

    axB = fig.add_subplot(gs[0, 1])
    axB.axvspan(11, 16, color="#b48a00", alpha=.12)
    axB.loglog(frq[mf], pr[mf], color="0.12", lw=2.6, label="real N2 PSD")
    axB.loglog(frq[mf], ps[mf], color=PINK, lw=1.5, ls="--", label="surrogate PSD")
    axB.set_xlim(1, 45); axB.set_xlabel("frequency (Hz)", fontsize=11); axB.set_ylabel("power", fontsize=11)
    axB.tick_params(labelsize=9.5)
    axB.legend(frameon=False, fontsize=10, loc="lower left"); axB.spines[["top", "right"]].set_visible(False)
    axB.text(-0.16, 1.04, "B", transform=axB.transAxes, fontweight="bold", fontsize=16)

    axC = fig.add_subplot(gs[0, 2])
    axC.plot(tw, TMPL, color="0.12", lw=2); axC.axvline(0, color="0.7", lw=.7, ls=":")
    axC.axhline(0, color="0.35", lw=1.0)
    axC.set_xlim(-1, 1.5); axC.set_xlabel("time re: neg peak (s)", fontsize=11); axC.set_ylabel("µV", fontsize=11)
    axC.tick_params(labelsize=9.5)
    axC.spines[["top", "right"]].set_visible(False)
    axC.text(-0.14, 1.04, "C", transform=axC.transAxes, fontweight="bold", fontsize=16)

    LABF = ["broadband 1–45 Hz", "high-band 10–45 Hz", "high-band 20–45 Hz"]
    for k, (cc, lab) in enumerate(zip(COLS, LABF)):
        a = fig.add_subplot(gs[1, k])
        a.plot(taus, mR[k], color=cc, lw=2.4, label="real KC")
        a.fill_between(taus, loR[k], hiR[k], color=cc, alpha=.20, lw=0)
        a.plot(tflo, mF[k], color=GREY, lw=1.8, ls=(0, (4, 2)), label="surrogate floor")
        a.fill_between(tflo, loF[k], hiF[k], color=GREY, alpha=.25, lw=0)
        a.axhline(0, color="0.35", lw=1.0, ls=":"); a.axvline(0, color="0.6", lw=1, ls="--")
        lo, hi = np.percentile(resid[:, k], [2.5, 97.5])
        a.set_title(f"{lab}\nresidual = {resid[:, k].mean():+.2f} [{lo:+.2f}, {hi:+.2f}]",
                    fontsize=11.5, color=cc, fontweight="bold", loc="center")
        a.set_xlim(-5, 5); a.set_xlabel("time re: KC (s)", fontsize=11.5)
        a.tick_params(labelsize=10)
        if k == 0:
            a.set_ylabel("Δ aperiodic exponent", fontsize=11.5)
        a.legend(frameon=False, fontsize=10.5, loc="upper right"); a.spines[["top", "right"]].set_visible(False)
        a.text(-0.16 if k == 0 else -0.1, 1.13, "DEF"[k], transform=a.transAxes,
               fontweight="bold", fontsize=16, color="0.1")
    out = cfg.FIG_DIR / "fig4_surrogate_floor.png"
    plt.savefig(out, dpi=190, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    main()
