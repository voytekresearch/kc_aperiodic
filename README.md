# Aperiodic dynamics track cortical state shifts during sleep K-complexes

Analysis code for the paper. All results and figures are reproduced from the
cleaned MASS SS2 recordings by a single driver script.

## What the paper reports, and where each number comes from

| Result | Analysis script | Figure |
| --- | --- | --- |
| Event-locked aperiodic exponent deflection across the fit-range ladder | `analysis/a01_event_locked.py` | `figures/fig2_ladder.py` |
| KC-peak vs N2-baseline spectra | `analysis/a01_event_locked.py` | `figures/fig2_ladder.py` (A) |
| KC-locked spectrogram | `analysis/a05_spectrogram.py` | `figures/fig2_ladder.py` (C) |
| Amplitude dissociation (r vs fit floor) | `analysis/a03_amplitude.py` | `figures/fig3_amplitude.py` |
| Spectrum-matched surrogate floor | `analysis/a02_surrogate_floor.py` | `figures/fig4_surrogate_floor.py` |
| Event-free surrogate (robustness) | `analysis/a02_surrogate_floor.py` | reported in `stats/stats.py` |
| Waveform-regression control | `analysis/a04_regression.py` | `figures/fig5_regression.py` |
| N2 baseline exponent | `analysis/a06_epoch_baseline.py` | reported in `stats/stats.py` |
| Expert-mark → negative-peak alignment | (computed in figure) | `figures/figS1_alignment.py` |
| All reported statistics (deflections, CIs, t-tests, residuals) | — | `stats/stats.py` |

## Layout

```
notebooks/            upstream steps that turn the raw MASS SS2 recordings into the cleaned data
  01_preprocessing.ipynb    per-subject: N2 extraction, filtering, ICA + ICLabel,
                            AutoReject, mastoid re-reference -> cleaned raw + annotations
  02_time_resolved.ipynb    time-resolved spectral parameterization (2 s window, 0.5 s step)
kcaperiodic/          importable package
  config.py           subjects, paths, all analysis parameters (single source of truth)
  core.py             data loading + every shared analysis function
analysis/             a00-a06: build caches in data/ (one .npz per subject)
figures/              fig2-fig5 + figS1: read data/, write figures_out/
stats/                stats.py: reads data/, prints table + writes stats_summary.json
run_all.py            end-to-end driver for the analysis (a00-a06 -> figures -> stats)
```

The two notebooks in `notebooks/` document how the cleaned data was produced from
the raw MASS SS2 EDFs; they are run once per subject upstream and are not part of
`run_all.py`. `01_preprocessing.ipynb` writes each subject's cleaned N2 recording
and expert annotations, which are placed under `DATA_DIR/<subject>/cleaned/` as
`<subject>_cleaned_raw.fif` and `<subject>_annotations.csv` for the analysis
scripts to load. They contain the original absolute paths used on the author's
machine; edit those to your own before running.

## Reproduce

1. Point `DATA_DIR` in `kcaperiodic/config.py` at the cleaned data. Each subject is
   expected at `DATA_DIR/<subject>/cleaned/<subject>_cleaned_raw.fif` with a matching
   `<subject>_annotations.csv` (expert KC and spindle onsets).
2. `pip install -r requirements.txt`
3. From the repo root:

   ```bash
   python run_all.py
   ```

   This builds the KC template, generates every per-subject cache in `data/`
   (skipping any already present), renders the figures into `figures_out/`, and
   prints the statistics. To regenerate a single subject's caches, delete its
   `data/*_<subject>.npz` files (or pass the id: `python run_all.py 01-02-0019`).

## Method notes

- **Two estimators, kept explicit.** The primary event-locked estimator is a 2 s
  Welch window (`nperseg` = 1 s), matched to the K-complex's low-frequency footprint.
  The per-event amplitude and regression controls use 1 s multitaper windows.
- **Fit-range ladder.** Every spectrum is parameterized with `specparam` (fixed
  aperiodic mode) over three ranges: 1–45, 10–45, and 20–45 Hz.
- **Spectrum-matched surrogate floor.** A phase-randomized Fourier surrogate
  preserves each subject's N2 power spectrum while destroying all time-locked
  structure; the amplitude-matched KC template is then injected and the pipeline
  re-run, giving a fair waveform-only floor.
- **Subjects.** All 19 MASS SS2 subjects (C3, expert 1) are included.

## Data availability

The MASS SS2 dataset is available from the Montreal Archive of Sleep Studies. The
cleaned recordings and expert annotations are not redistributed here. The cached
per-subject intermediates in `data/` are derived from those recordings and are
likewise not committed (they are git-ignored); running `run_all.py` regenerates
them locally. The rendered figures in `figures_out/` are the only aggregate
outputs kept in the repository.
