"""
Central configuration for the K-complex aperiodic analysis.

All parameters, subject list, and paths live here so every script is
reproducible from a single source of truth. Edit DATA_DIR to point at the
cleaned MASS SS2 data on your machine.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
# Cleaned data: <DATA_DIR>/<subject>/cleaned/<subject>_cleaned_raw.fif (+ _annotations.csv)
# Override with the KC_DATA_DIR environment variable if the data lives elsewhere.
DATA_DIR = Path(os.environ.get("KC_DATA_DIR", "~/Desktop/SS2_Results/New")).expanduser()
_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / "data"            # cached per-subject intermediates (.npz)
FIG_DIR = _ROOT / "figures_out"       # rendered figures (.png)
CACHE_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- subjects
# MASS SS2, C3, expert 1. All 19 subjects are included; 01-02-0019 was
# previously excluded but retains 191 usable K-complexes after preprocessing.
SUBJECTS = [f"01-02-{i:04d}" for i in range(1, 20)]

# ---------------------------------------------------------------- signal / spectral params
FS = 256                              # sampling rate (Hz)

# Fit ranges (the "fit-range ladder")
FRANGES = [(1.0, 45.0), (10.0, 45.0), (20.0, 45.0)]   # broadband, high-band, 20-45
BAND_LABELS = ["1-45", "10-45", "20-45"]

# specparam settings, held constant everywhere
SPECPARAM_KW = dict(peak_width_limits=[1, 12], max_n_peaks=5,
                    peak_threshold=2, aperiodic_mode="fixed", verbose=False)

# Primary event-locked estimator: 2 s Welch windows (Welch nperseg = 1 s),
# matched to the K-complex's low-frequency footprint (Cellier et al., 2026, Rec. 5).
LWIN = 2.0                            # event-locked window length (s)
NPERSEG_SEC = 1.0                     # Welch sub-window inside each LWIN (s)

# Amplitude / regression control estimator: 1 s multitaper windows.
WIN_MT = 1.0
MT_BANDWIDTH = 4.0
R2_THRESH = 0.9

# Event-locked sampling
BASELINE_S = 5.0                      # +/- baseline windows for the deflection metric
BASE_ABS_T = 3.0                      # |t| > this counts as far-field baseline

# Number of KCs per subject to include (None = all). Kept modest for runtime;
# set to None to use every annotated KC.
MAX_EVENTS = 120

# Reproducibility
SEED = 0
N_BOOT = 10000                        # bootstrap resamples
