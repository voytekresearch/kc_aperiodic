"""Build the grand-average isolated-KC template (used by the floor and
regression analyses) and cache it to data/template.npz."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from kcaperiodic import config as cfg
from kcaperiodic import core

OUT = cfg.CACHE_DIR / "template.npz"


def main():
    segs = []
    for sid in cfg.SUBJECTS:
        d = core.load_subject(sid, cfg.DATA_DIR)
        t = core.build_kc_template(d["sig"], d["fs"], d["kc_onsets"], d["spindle_onsets"])
        segs.append(t)
    template = np.mean(segs, 0)
    np.savez(OUT, template=template, n_subjects=len(segs))
    print(f"template.npz: len={len(template)}, neg peak at {int(np.argmin(template))}, "
          f"depth {-template.min():.1f} uV, from {len(segs)} subjects")


if __name__ == "__main__":
    main()
