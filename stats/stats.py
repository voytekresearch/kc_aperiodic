"""Compute every statistic reported in the paper from the cached intermediates,
at whatever n is present in data/. Prints a labelled table and writes
stats/stats_summary.json.

Run after all analysis scripts have populated data/.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import glob
import json
import numpy as np
from scipy.stats import pearsonr, ttest_1samp, ttest_rel
from kcaperiodic import config as cfg

C = str(cfg.CACHE_DIR) + "/"
BANDS = cfg.BAND_LABELS
rng = np.random.default_rng(cfg.SEED)


def boot_ci(x, nboot=cfg.N_BOOT):
    x = np.asarray(x); idx = rng.integers(0, len(x), (nboot, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def peak_defl(mean, taus):
    """Peak-of-trace deflection: max over tau minus far-field baseline (|t|>3)."""
    base = np.abs(taus) > 3
    return np.array([np.nanmax(mean[k]) - np.nanmean(mean[k][base]) for k in range(len(BANDS))])


def t0_defl(mean, taus):
    """Fixed-t=0 deflection: value at nearest tau to 0 minus far-field baseline."""
    base = np.abs(taus) > 3; i0 = int(np.argmin(np.abs(taus)))
    return np.array([mean[k][i0] - np.nanmean(mean[k][base]) for k in range(len(BANDS))])


def main():
    out = {}

    # ---- event-locked exponent deflection (2 s Welch) ----
    FO = sorted(glob.glob(C + "evtrace_*.npz")); taus = np.load(FO[0])["taus"]
    REAL = np.array([np.load(f)["mean"] for f in FO]); n = len(FO)
    out["n_subjects"] = n
    peak = np.array([peak_defl(m, taus) for m in REAL])   # (n, 3)
    t0 = np.array([t0_defl(m, taus) for m in REAL])
    out["event_locked"] = {}
    for k, b in enumerate(BANDS):
        m, lo, hi = boot_ci(peak[:, k]); t, p = ttest_1samp(peak[:, k], 0)
        m0, lo0, hi0 = boot_ci(t0[:, k])
        out["event_locked"][b] = dict(peak_defl=m, peak_ci=[lo, hi],
                                      t0_defl=m0, t0_ci=[lo0, hi0],
                                      t=float(t), p=float(p))

    # ---- surrogate floor residual (real peak minus floor peak) ----
    S = [f.split("floor_")[1][:10] for f in sorted(glob.glob(C + "floor_*.npz"))
         if "floor_ef_" not in f]
    if S:
        tflo = np.load(C + f"floor_{S[0]}.npz")["taus"]
        Rr = np.array([np.load(C + f"evtrace_{s}.npz")["mean"] for s in S])
        Ff = np.array([np.load(C + f"floor_{s}.npz")["mean"] for s in S])
        resid = np.zeros((cfg.N_BOOT, 3))
        for i in range(cfg.N_BOOT):
            idx = rng.integers(0, len(S), len(S))
            resid[i] = peak_defl(np.nanmean(Rr[idx], 0), taus) - peak_defl(np.nanmean(Ff[idx], 0), tflo)
        out["surrogate_floor"] = {b: dict(residual=float(resid[:, k].mean()),
                                          ci=[float(np.percentile(resid[:, k], 2.5)),
                                              float(np.percentile(resid[:, k], 97.5))])
                                  for k, b in enumerate(BANDS)}

    # ---- event-free surrogate robustness ----
    Se = [f.split("floor_ef_")[1][:10] for f in sorted(glob.glob(C + "floor_ef_*.npz"))]
    if Se:
        tfe = np.load(C + f"floor_ef_{Se[0]}.npz")["taus"]
        Fe = np.array([np.load(C + f"floor_ef_{s}.npz")["mean"] for s in Se])
        Re = np.array([np.load(C + f"evtrace_{s}.npz")["mean"] for s in Se])
        rr = np.zeros((cfg.N_BOOT, 3))
        for i in range(cfg.N_BOOT):
            idx = rng.integers(0, len(Se), len(Se))
            rr[i] = peak_defl(np.nanmean(Re[idx], 0), taus) - peak_defl(np.nanmean(Fe[idx], 0), tfe)
        out["surrogate_floor_eventfree"] = {b: dict(residual=float(rr[:, k].mean()),
                                                    ci=[float(np.percentile(rr[:, k], 2.5)),
                                                        float(np.percentile(rr[:, k], 97.5))])
                                            for k, b in enumerate(BANDS)}

    # ---- amplitude coupling (per-band r + paired tests) ----
    FA = sorted(glob.glob(C + "ampdefl_*.npz")); R = []
    for f in FA:
        z = np.load(f); a = z["amp"]; d = z["defl"]; ok = ~np.isnan(a)
        R.append([pearsonr(a[ok], d[ok, k])[0] for k in range(3)])
    R = np.array(R)
    out["amplitude"] = {}
    for k, b in enumerate(BANDS):
        t, p = ttest_1samp(R[:, k], 0)
        out["amplitude"][b] = dict(r_mean=float(R[:, k].mean()), r_sd=float(R[:, k].std(ddof=1)),
                                   t=float(t), p=float(p))
    out["amplitude"]["monotonic"] = int(np.sum(np.all(np.diff(R, axis=1) < 0, axis=1)))
    out["amplitude"]["paired_1v10"] = float(ttest_rel(R[:, 0], R[:, 1]).pvalue)
    out["amplitude"]["paired_10v20"] = float(ttest_rel(R[:, 1], R[:, 2]).pvalue)

    # ---- waveform regression (raw vs residual event-locked peak) ----
    FR = sorted(glob.glob(C + "regtr_*.npz"))
    if FR:
        rtaus = np.load(FR[0])["taus"]
        Eo = np.array([np.load(f)["Eo"] for f in FR]); Er = np.array([np.load(f)["Er"] for f in FR])
        raw = np.array([peak_defl(m, rtaus) for m in Eo]); res = np.array([peak_defl(m, rtaus) for m in Er])
        out["regression"] = {}
        for k, b in enumerate(BANDS):
            mr, lr, hr = boot_ci(raw[:, k]); ms, ls, hs = boot_ci(res[:, k])
            t, p = ttest_1samp(res[:, k], 0)
            out["regression"][b] = dict(raw=mr, raw_ci=[lr, hr], residual=ms, residual_ci=[ls, hs],
                                        residual_t=float(t), residual_p=float(p))

    # ---- N2 baseline exponent ----
    FB = sorted(glob.glob(C + "epochbase_*.npz"))
    if FB:
        B = np.array([np.load(f)["B"] for f in FB])
        out["baseline_exponent"] = {b: dict(mean=float(B[:, k].mean()), sd=float(B[:, k].std(ddof=1)))
                                    for k, b in enumerate(BANDS)}

    # ---- print ----
    print(f"\n=== n = {n} subjects ===\n")
    print("EVENT-LOCKED exponent deflection (2 s Welch)")
    for b in BANDS:
        e = out["event_locked"][b]
        print(f"  {b:>6}: peak Δχ={e['peak_defl']:+.2f} [{e['peak_ci'][0]:+.2f},{e['peak_ci'][1]:+.2f}]"
              f"  t0 Δχ={e['t0_defl']:+.2f}  t={e['t']:.1f} p={e['p']:.1e}")
    if "surrogate_floor" in out:
        print("\nSURROGATE-FLOOR residual (real minus floor)")
        for b in BANDS:
            s = out["surrogate_floor"][b]
            print(f"  {b:>6}: {s['residual']:+.2f} [{s['ci'][0]:+.2f},{s['ci'][1]:+.2f}]")
    if "surrogate_floor_eventfree" in out:
        print("\nEVENT-FREE surrogate residual (robustness)")
        for b in BANDS:
            s = out["surrogate_floor_eventfree"][b]
            print(f"  {b:>6}: {s['residual']:+.2f} [{s['ci'][0]:+.2f},{s['ci'][1]:+.2f}]")
    print("\nAMPLITUDE coupling (within-subject r)")
    for b in BANDS:
        a = out["amplitude"][b]
        print(f"  {b:>6}: r={a['r_mean']:+.2f} ± {a['r_sd']:.2f}  t={a['t']:.1f} p={a['p']:.1e}")
    print(f"  monotonic decrease: {out['amplitude']['monotonic']}/{n}"
          f"  (1-45>10-45 p={out['amplitude']['paired_1v10']:.0e},"
          f" 10-45>20-45 p={out['amplitude']['paired_10v20']:.0e})")
    if "regression" in out:
        print("\nWAVEFORM REGRESSION (event-locked peak, raw -> residual)")
        for b in BANDS:
            r = out["regression"][b]
            print(f"  {b:>6}: {r['raw']:+.2f} -> {r['residual']:+.2f} "
                  f"[{r['residual_ci'][0]:+.2f},{r['residual_ci'][1]:+.2f}]  p={r['residual_p']:.1e}")
    if "baseline_exponent" in out:
        print("\nN2 BASELINE exponent")
        for b in BANDS:
            z = out["baseline_exponent"][b]
            print(f"  {b:>6}: χ={z['mean']:.2f} ± {z['sd']:.2f}")

    with open(cfg.CACHE_DIR.parent / "stats" / "stats_summary.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote stats/stats_summary.json")


if __name__ == "__main__":
    main()
