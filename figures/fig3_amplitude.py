"""Figure 3 -- amplitude dissociation across the fit-range ladder.
A-C: per-event KC amplitude vs exponent deflection, one panel per fit range.
D: within-subject correlation r across the three fit ranges (paired).
E: r as a continuous function of the lower fit bound Fmin.
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, ttest_rel
from _style import COLS, BLUE, TEAL, GOLD
from kcaperiodic import config as cfg

C = str(cfg.CACHE_DIR) + "/"
LAB = ["broadband (1–45 Hz)", "high-band (10–45 Hz)", "high-band (20–45 Hz)"]


def main():
    # per-event amplitude & deflection, 3 bands
    F = sorted(glob.glob(C + "ampdefl_*.npz"))
    A_c = [[], [], []]; D_c = [[], [], []]; R = []
    for f in F:
        z = np.load(f); a = z["amp"]; d = z["defl"]; ok = ~np.isnan(a)
        a = a[ok]; d = d[ok]
        R.append([pearsonr(a, d[:, k])[0] for k in range(3)])
        for k in range(3):
            A_c[k].append(a - a.mean()); D_c[k].append(d[:, k] - d[:, k].mean())
    R = np.array(R)
    A_c = [np.concatenate(x) for x in A_c]; D_c = [np.concatenate(x) for x in D_c]

    # r vs fit floor
    FR = sorted(glob.glob(C + "rfmin_*.npz")); fmins = np.load(FR[0])["fmins"]
    Rf = []
    for f in FR:
        z = np.load(f); a = z["amp"]; D = z["defl"]; ok = ~np.isnan(a)
        Rf.append([pearsonr(a[ok], D[ok, k])[0] for k in range(len(fmins))])
    Rf = np.array(Rf); rf_m = Rf.mean(0); rf_se = Rf.std(0, ddof=1) / np.sqrt(len(FR))

    fig = plt.figure(figsize=(14.5, 8.4))
    gs = GridSpec(2, 3, hspace=.34, wspace=.30, left=.06, right=.975, top=.93, bottom=.08)
    xl = (-95, 115); yl = (-1.7, 1.9)

    def scatter(ax, k, letter):
        ax.scatter(A_c[k], D_c[k], s=6, alpha=.10, color=COLS[k], edgecolors="none", rasterized=True)
        x = np.array(xl); ax.plot(x, np.polyval(np.polyfit(A_c[k], D_c[k], 1), x), color=COLS[k], lw=2.6)
        ax.axhline(0, color="0.8", lw=.8); ax.set_xlim(*xl); ax.set_ylim(*yl)
        r = R[:, k]
        ax.text(.04, .95, f"{LAB[k]}\nr = {r.mean():+.2f} ± {r.std(ddof=1):.2f}",
                transform=ax.transAxes, va="top", fontweight="bold", color=COLS[k], fontsize=10.5)
        ax.set_xlabel("KC amplitude (µV, within-subj centered)", fontsize=9.5)
        if k == 0:
            ax.set_ylabel("Δ aperiodic exponent", fontsize=10)
        ax.set_title(letter, loc="left", fontweight="bold", fontsize=13)
        ax.spines[["top", "right"]].set_visible(False)

    for k, L in zip(range(3), "ABC"):
        scatter(fig.add_subplot(gs[0, k]), k, L)

    # D: paired slopegraph across the 3 bands
    axD = fig.add_subplot(gs[1, 0]); xs = [0, 1, 2]
    for i in range(len(R)):
        axD.plot(xs, R[i], color="0.78", lw=.8, zorder=1)
    for k in range(3):
        axD.scatter([k] * len(R), R[:, k], s=32, color=COLS[k], edgecolors="white", lw=.6, zorder=3)
        axD.plot([k - .2, k + .2], [R[:, k].mean()] * 2, color=COLS[k], lw=3.2, zorder=4)
    axD.axhline(0, color="0.8", lw=.8); axD.set_xticks(xs)
    axD.set_xticklabels(["1–45", "10–45", "20–45"]); axD.set_xlim(-.4, 2.4)
    axD.set_ylabel("within-subject r\n(KC amplitude vs Δ exponent)", fontsize=10)
    axD.set_xlabel("fit range (Hz)", fontsize=9.5)
    axD.set_title("D", loc="left", fontweight="bold", fontsize=13)
    axD.spines[["top", "right"]].set_visible(False)
    p_bh = ttest_rel(R[:, 0], R[:, 1]).pvalue; p_h2 = ttest_rel(R[:, 1], R[:, 2]).pvalue
    mono = int(np.sum(np.all(np.diff(R, axis=1) < 0, axis=1)))
    print(f"panel D: {mono}/{len(R)} monotonic; 1-45>10-45 p={p_bh:.0e}, 10-45>20-45 p={p_h2:.0e}")

    # E: r vs fit floor (continuous)
    axE = fig.add_subplot(gs[1, 1:])
    axE.axhline(0, color="0.8", lw=.9)
    axE.fill_between(fmins, rf_m - rf_se, rf_m + rf_se, color="0.75", alpha=.35)
    axE.plot(fmins, rf_m, "-", color="0.25", lw=2.2, zorder=2)
    for fm, c in [(1, BLUE), (10, TEAL), (20, GOLD)]:
        i = int(np.argmin(np.abs(fmins - fm)))
        axE.plot(fm, rf_m[i], "o", color=c, ms=13, zorder=4, mec="white", mew=1)
        axE.annotate(f"{fm}–45  (r={rf_m[i]:+.2f})", (fm, rf_m[i]), textcoords="offset points",
                     xytext=(8, 10 if fm < 20 else -16), color=c, fontweight="bold", fontsize=9.5)
    axE.set_xlabel("lower fit bound  Fmin (Hz)", fontsize=10)
    axE.set_ylabel("mean within-subject r  (KC amplitude vs Δ exponent)", fontsize=10)
    axE.set_title("E", loc="left", fontweight="bold", fontsize=13)
    axE.set_xlim(0.5, 20.5); axE.spines[["top", "right"]].set_visible(False)

    out = cfg.FIG_DIR / "fig3_amplitude.png"
    plt.savefig(out, dpi=190, bbox_inches="tight")
    print("saved", out, "; r =", [round(R[:, k].mean(), 3) for k in range(3)])


if __name__ == "__main__":
    main()
