"""Supplementary figure S1 -- expert-mark to negative-peak alignment.
A: KC epochs locked to the expert onset (example subject).
B: same epochs re-locked to the detected negative peak.
C: distribution of expert-mark -> negative-peak latencies across subjects.
"""
import numpy as np
import matplotlib.pyplot as plt
from _style import BLUE
from kcaperiodic import config as cfg
from kcaperiodic import core

FS = cfg.FS
GREY = "0.6"
half = int(1.5 * FS); tt = np.arange(-half, half) / FS
SEARCH = int(1.5 * FS)


def negpeak(sig, onset_sample):
    a = int(onset_sample); b = min(a + SEARCH, len(sig))
    if b - a < 10:
        return None
    j = a + int(np.argmin(sig[a:b]))
    if j <= a + int(0.03 * FS) or j >= b - int(0.03 * FS):
        return None
    return j


def epochs(sig, centers):
    E = []
    for c in centers:
        c = int(c)
        if c - half < 0 or c + half > len(sig):
            continue
        E.append(sig[c - half:c + half] - np.mean(sig[c - half:c - half + int(0.3 * FS)]))
    return np.array(E)


def main():
    d = core.load_subject("01-02-0002", cfg.DATA_DIR); sig = d["sig"]
    onsets = [int(round(o * FS)) for o in d["kc_onsets"]]
    pairs = [(o, negpeak(sig, o)) for o in onsets]
    pairs = [(o, p) for o, p in pairs
             if p is not None and o - half >= 0 and p + half < len(sig) and o + half < len(sig)]
    sel = pairs[:15]
    on_ep = epochs(sig, [o for o, _ in sel]); pk_ep = epochs(sig, [p for _, p in sel])
    lat = [(p - o) / FS for o, p in sel]

    offs = []
    for s in cfg.SUBJECTS[:6]:
        dd = core.load_subject(s, cfg.DATA_DIR); sg = dd["sig"]
        for o in dd["kc_onsets"]:
            p = negpeak(sg, int(round(o * FS)))
            if p is not None:
                offs.append(p / FS - o)
    offs = np.array(offs); iqr = np.percentile(offs, [25, 75]) * 1000; sd = offs.std() * 1000

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for e in on_ep:
        ax[0].plot(tt, e, color=GREY, lw=.6, alpha=.6)
    ax[0].plot(tt, on_ep.mean(0), color=BLUE, lw=2.8)
    for l, e in zip(lat, on_ep):
        ax[0].plot(l, e[np.argmin(np.abs(tt - l))], 'v', color="#c0392b", ms=5)
    ax[0].axvline(0, color="k", lw=1, ls="--"); ax[0].set_xlim(-1, 1.5); ax[0].set_ylim(-175, 140)
    ax[0].set_title("A", loc="left", fontweight="bold", fontsize=14)
    ax[0].set_xlabel("time re: expert mark (s)"); ax[0].set_ylabel("µV")
    ax[0].spines[["top", "right"]].set_visible(False)

    for e in pk_ep:
        ax[1].plot(tt, e, color=GREY, lw=.6, alpha=.6)
    ax[1].plot(tt, pk_ep.mean(0), color=BLUE, lw=2.8)
    ax[1].axvline(0, color="k", lw=1, ls="--"); ax[1].set_xlim(-1, 1.5); ax[1].set_ylim(-175, 140)
    ax[1].set_title("B", loc="left", fontweight="bold", fontsize=14)
    ax[1].set_xlabel("time re: negative peak (s)"); ax[1].spines[["top", "right"]].set_visible(False)

    ax[2].hist(offs * 1000, bins=28, color=BLUE, alpha=.85)
    ax[2].set_title("C", loc="left", fontweight="bold", fontsize=14)
    ax[2].set_xlabel("expert mark → negative peak (ms)"); ax[2].set_ylabel("count")
    ax[2].spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = cfg.FIG_DIR / "figS1_alignment.png"
    plt.savefig(out, dpi=190, bbox_inches="tight")
    print(f"saved {out}  n={len(offs)} median={np.median(offs) * 1000:.0f} "
          f"SD={sd:.0f} IQR={iqr[0]:.0f}-{iqr[1]:.0f} ms")


if __name__ == "__main__":
    main()
