"""Shared helper: run a per-subject analysis for one subject (CLI arg) or all.

Importing this module also puts the repo root on sys.path, so the analysis
scripts can `from kcaperiodic import ...` when run directly from the repo root
(e.g. `python analysis/a01_event_locked.py`).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kcaperiodic.config import SUBJECTS  # noqa: E402


def run(per_subject_fn):
    """per_subject_fn(subject_id) -> None. Runs the subjects passed on the command
    line, otherwise every subject in SUBJECTS (per_subject_fn skips cached ones)."""
    subs = sys.argv[1:] if len(sys.argv) > 1 else SUBJECTS
    for s in subs:
        per_subject_fn(s)
