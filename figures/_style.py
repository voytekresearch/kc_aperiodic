"""Shared figure styling and the band color scheme used across all figures."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

# make the package importable when a figure script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BLUE, TEAL, GOLD = "#2c5aa0", "#1f9e8b", "#d98a1f"   # 1-45, 10-45, 20-45
GREY = "#8a8f98"                                     # surrogate floor
PINK = "#d6336c"                                     # regression residual / surrogate
COLS = [BLUE, TEAL, GOLD]
BAND_LAB = ["1–45 Hz", "10–45 Hz", "20–45 Hz"]
