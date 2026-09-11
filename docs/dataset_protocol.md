# Dataset Protocol

> Historical/supporting note. For current terminology and experiment status,
> see [research/current_state.md](research/current_state.md). This document
> does not establish completed experiments.

## 1. Input Specifications
- **Input Resolution:** `4096 × 4096`
- **Number of Channels:** `13`
- **Task:** Solar flare forecasting

## 2. Forecasting Targets
- **Event Thresholds:**
  - `≥ C1.0`
  - `≥ M1.0`
  - `≥ X1.0`
- **Forecast Horizons:**
  - `2 hours`
  - `24 hours`
  - *(Additional horizons (e.g., 6h, 12h) only if computationally practical)*

## 3. Dataset Splits
The index files are supplied separately from this repository. Subdirectories
categorize them by threshold and forecasting window (for example, `C24w` for a
C-class 24-hour window and `M2w` for an M-class 2-hour window).

The splits are provided as pre-computed CSV files:
- `train_{Config}.csv`
- `val_{Config}.csv`
- `test_{Config}.csv`
- `leaky_val_{Config}.csv` (Note: requires audit to understand why it is marked leaky)

## 4. Sampling Strategy
- For limited data experiments (learning curves), we will use the pre-computed frequency-sampled training files (e.g., `train_C24w_freq2.csv`, `freq4`, `freq8`, `freq12`, `freq24`), which correspond to fractional subsets of the full dataset.
