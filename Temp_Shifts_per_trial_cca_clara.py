# %%
# -*- coding: utf-8 -*-
"""
Per-trial Canonical Correlation Analysis (CCA)
=============================================

For every *single* trial, we compute the **maximal canonical correlation**
between the multichannel EEG (theta-band power) and pupil-diameter time
series, allowing for small temporal lags (±1 s in 100 ms steps).

Changes compared to `Temp_Shifts_per_trial_clara.py`
----------------------------------------------------
* **CCA instead of per-electrode Pearson correlations**.
* Preserves the original normalisation strategy:
  - EEG: centre each channel and scale so that the *total* variance across
    **time x channels** equals 1 ↝ keeps relative channel amplitudes.
  - Pupil: trial-wise z-score (mean 0, sd 1).
* Saves one row per *(subject, condition, load, epoch)* with the best
  correlation and lag.

Output
------
`trial_level_cca.csv`  -  columns:
    subject, condition (memory/control), load (05/09/13), epoch (file name),
    r_max (canonical corr.), lag_ms (best EEG→pupil lag, ms)

Requires
--------
* scikit-learn ≥ 1.0  (for `sklearn.cross_decomposition.CCA`)
* pandas, numpy
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel


# %%
###############################################################################
# User settings
###############################################################################
PUPIL_ROOT = Path(r"C:\Users\cdd\Documents\Uni\Special_course\pupil_processed_clara")
EEG_ROOT   = Path(r"C:\Users\cdd\Documents\Uni\Special_course\code\Special_course\eeg_theta_processed2")
OUT_FILE   = Path("trial_level_cca.csv")

SUBJECTS = np.setdiff1d(np.arange(32, 98), [32, 37, 53, 61, 66, 78, 84, 94, 96])
FRONTAL_MIDLINE = ['AFz','AF3','AF4','Fz','F1','F2','F3','F4','FC3','FC1','FC2','FC4','Cz','C3','C1','C2','C4']

SHIFTS = np.arange(-100, 101)   # ±1 s @100 Hz  →  10ms increments
WIN_OFFSET = 110               # discard first/last 110 ms for padding safety
###############################################################################


def normalise_eeg(matrix: np.ndarray) -> np.ndarray:
    """Centre each channel and scale so \sum_{t,c} x² = 1."""
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    var_total = np.mean(centred ** 2)
    return centred / np.sqrt(var_total)


def best_cca_corr(eeg: np.ndarray, pupil: np.ndarray) -> tuple[float, int]:
    """Return (r_max, best_lag_ms) for one trial."""
    best_r, best_shift = -np.inf, 0

    # window of interest (to avoid circular‑shift artefacts)
    T = min(len(pupil), len(eeg))
    win = slice(WIN_OFFSET, T - WIN_OFFSET)

    # z‑score pupil once outside the loop
    pupil_z = (pupil - pupil.mean()) / pupil.std(ddof=0)
    pupil_z = pupil_z[win].reshape(-1, 1)

    for s in SHIFTS:
        eeg_shifted = np.roll(eeg, s, axis=0)[win]

        # CCA with 1 component (degenerates to optimal regression)
        cca = CCA(n_components=1, max_iter=1000)
        cca.fit(eeg_shifted, pupil_z)

        #c_weights_eeg, c_weights_pupil = cca.x_weights_.flatten(), cca.y_weights_.flatten()
        #combined_eeg = eeg_shifted @ c_weights_eeg
        #combined_pupil = pupil_z * c_weights_pupil # Since pupil has 1D, just multiply

        # Calculate peraron correlation between the combinaed EEG and pupil data
        #r = np.corrcoef(combined_eeg, combined_pupil.flatten())[0, 1]
        u, v = cca.transform(eeg_shifted, pupil_z)
        r = np.corrcoef(u[:, 0], v[:, 0])[0, 1]

        if r > best_r:
            best_r, best_shift = r, s * 10  # samples → milliseconds

    return float(best_r), int(best_shift)

# %%
###############################################################################
# Main loop
###############################################################################
results = []

for sub in SUBJECTS:
    sub_tag = f"sub-{sub:03d}"
    eeg_sub = EEG_ROOT / sub_tag
    if not eeg_sub.exists():
        print(f"{sub_tag}: EEG folder missing - skipped")
        continue
    print(f"Processing {sub_tag}...")

    for cond_path in eeg_sub.iterdir():          # memory / control
        for load_path in cond_path.iterdir():    # 05 / 09 / 13
            eeg_epochs  = sorted(load_path.glob("trial_*.csv"))
            pupil_path  = PUPIL_ROOT / sub_tag / cond_path.name / load_path.name
            
            if not pupil_path.exists():
                print(f"Skipping {pupil_path} as it does not exist.")
                continue

            pupil_epochs = sorted(pupil_path.glob("trial_*.csv"))

            common = {e.name for e in eeg_epochs} & {p.name for p in pupil_epochs}
            if not common:
                print(f"{sub_tag} {cond_path.name} {load_path.name}: no common epochs - skipped")
                continue
            print(f"{sub_tag} {cond_path.name} {load_path.name}: {len(common)} common epochs")

            for fname in common:
                eeg_df   = pd.read_csv(load_path / fname, comment='#', skip_blank_lines=True, index_col=0)
                pupil_df = pd.read_csv(pupil_path / fname, comment='#', names=['time','diameter_z'], index_col=0)

                eeg = normalise_eeg(eeg_df.values)                      # (T,C)
                pupil = pupil_df['diameter_z'].values.astype(float)     # (T,)

                if len(pupil) < 40 or abs(len(pupil) - len(eeg)) > 30:
                    print(f"{sub_tag} {cond_path.name} {load_path.name} {fname}: invalid trial - skipped")
                    continue  # skip pathological trials

                r_max, lag_ms = best_cca_corr(eeg, pupil)
                results.append({
                    'subject': sub_tag,
                    'condition': cond_path.name,
                    'load': int(load_path.name),
                    'epoch': fname,
                    'r_max': r_max,
                    'lag_ms': lag_ms
                })

# -------------------------------------------------------------------------
print(f"Finished – valid trials: {len(results)}  →  saving {OUT_FILE}")
pd.DataFrame(results).to_csv(OUT_FILE, index=False)


df = pd.read_csv("trial_level_cca.csv")

# subject-wise means
pivot = (df.groupby(["subject", "condition"])["r_max"]
           .mean()
           .unstack())           # columns should be: control | memory
paired = pivot.dropna()

# Check the table
print(paired.round(3))           # sanity check

# Difference in the expected direction: memory − control
diff = paired["memory"] - paired["control"]

# Paired t-test (two-tailed)
t2, p2 = ttest_rel(paired["memory"], paired["control"])

# Convert to one-tailed since H1: memory > control
t_one_sided = t2
p_one_sided = p2 / 2 if diff.mean() > 0 else 1 - p2 / 2

print(f"\nMean difference (memory − control): {diff.mean():.3f}")
print(f"Two-tailed p = {p2:.4f}")
print(f"One-tailed p (memory > control) = {p_one_sided:.4f}")