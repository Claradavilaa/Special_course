# -*- coding: utf-8 -*-
"""fig5c_replication.py — exact theta‑band remake of Kosachenko et al. 2023 Fig 5c
================================================================================
Usage
-----
1.  Edit `BASE_DIR` so it points to the folder that contains the sub‑folders
    `sub-001`, `sub-002`, … with the *.set* files.
2.  Adapt the `SUBJECTS` list if you do not want to process every participant.
3.  Run the script (terminal or Jupyter cell) — a figure pops up with the
    red *Memory* and blue *Control* curves.

Notes
-----
* Each *.set* file already contains **all conditions**; you only need the one
  path.
* Annotation codes are structured as **CC PP LL [R]**
      CC  – `50` control, `60` memory
      PP  – serial position of the *digit* (01‥13)
      LL  – list length of the *trial* (05, 09, 13)
      R   – (recall success 0/1, memory only)
* We create **one trial‑long epoch per list** (−2 s ➜ +11 / 17 / 27 s) and use
  a *single* baseline (−2…−1 s) as in the paper.  The time‑frequency transform
  is computed on the full 1‑45 Hz grid and collapsed to theta (4‑8 Hz) only
  afterwards.
* Empty digit positions (a 5‑digit list has no bins ≥ 6) are filled with `NaN`
  so they do not skew the grand average.
"""

import os
import numpy as np
import mne
from scipy.stats import sem
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# USER SETTINGS --------------------------------------------------------
# ---------------------------------------------------------------------
BASE_DIR  = r"C:\Users\cdd\Documents\Uni\Special_course\ds003838-download"
# SUBJECTS = np.arange(32, 34, 1)
SUBJECTS   = np.setdiff1d(np.arange(32, 99), [37, 53, 66, 94, 96])
#SUBJECTS   = np.setdiff1d(np.arange(32, 60), [37])
#ROI       = ['AFz','AF3','AF4','Fz','F1','F2','F3','F4','FC3','FC1','FC2','FC4','Cz','C3','C1','C2','C4']
ROI = ['Fz']
RESAMPLE  = 250                       # Hz; speeds up the TFR ∼4×

# ---------------------------------------------------------------------
# CONSTANTS ------------------------------------------------------------
# ---------------------------------------------------------------------
FREQS     = np.arange(1, 46)                                # 1‥45 Hz
N_CYCLES  = np.logspace(np.log10(3), np.log10(12), len(FREQS))
BASELINE  = (-1.0, 0.0)                                    # % change
BIN_DUR   = 2.0                                             # seconds / digit
LISTLEN_MAP = {5: 11.0, 9: 17.0, 13: 27.0}                  # tmax per load
X_POS     = np.arange(1, 14)                                # 1‥13 for plotting

# ---------------------------------------------------------------------
# HELPERS --------------------------------------------------------------
# ---------------------------------------------------------------------

def _theta_power(tfr, beg, end):
    """Return mean theta (4‑8 Hz) power (%) in the [beg, end] time‑window."""
    f_sel = np.where((FREQS >= 4) & (FREQS <= 8))[0]
    return tfr.copy().crop(beg, end).data[:, f_sel, :].mean()


def _get_first_digit_ids(ids, cond_prefix, list_len):
    """Return dict {code: id} for events that mark the *first* digit.
    * cond_prefix ∈ {'50', '60'}
    * list_len    ∈ {5, 9, 13}
    """
    list_code = f"{list_len:02d}"
    return {
        code: eid for code, eid in ids.items()
        if code.startswith(cond_prefix)          # condition
        and code[2:4] == '01'                    # serial position = 01
        and code[4:6] == list_code               # list length     = 05/09/13
    }


# ---------------------------------------------------------------------
# MAIN ----------------------------------------------------------------
# ---------------------------------------------------------------------
mem_all, ctrl_all = [], []                            # rows: subjects

for subj in SUBJECTS:
    raw_path = os.path.join(BASE_DIR,
                            f"sub-{subj:03d}", "eeg",
                            f"sub-{subj:03d}_task-memory_eeg.set")
    if not os.path.exists(raw_path):
        continue  # subject missing

    print(f"▶  Subject {subj:03d}")
    raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose='error')
    raw.filter(1., 45., fir_design='firwin', verbose='error')
    raw.set_eeg_reference('average', verbose='error')
    raw.resample(RESAMPLE, verbose='error')

    events, ids = mne.events_from_annotations(raw)

    # containers → one list of trial‑averaged values *per* digit position
    mem_by_pos  = [[] for _ in range(13)]
    ctrl_by_pos = [[] for _ in range(13)]

    # iterate over list lengths ------------------------------------------------
    for n_digits, tmax in LISTLEN_MAP.items():
        for cond_prefix, bucket in [('60', mem_by_pos), ('50', ctrl_by_pos)]:
            first_ids = _get_first_digit_ids(ids, cond_prefix, n_digits)
            if not first_ids:
                continue  # this subject has no such trials

            ep = mne.Epochs(
                raw, events, event_id=first_ids,
                tmin=-2.0, tmax=tmax,
                baseline=None, preload=True, picks='eeg', verbose='error')
            
            ep_csd = mne.preprocessing.compute_current_source_density(
                ep, # Copy the epochs to avoid modifying the original data
                sphere='auto',
                lambda2 = 1e-5,
                stiffness=4,
                n_legendre_terms=50,
                copy=True,
                verbose=True
            )

            # full 1‑45 Hz grid
            tfr = mne.time_frequency.tfr_morlet(
                ep_csd.pick_channels(ROI),
                freqs=FREQS, n_cycles=N_CYCLES,
                return_itc=False, verbose='error')
            tfr.apply_baseline(BASELINE, mode='percent')

            # slice out each real digit bin -----------------------------------
            for pos in range(1, n_digits + 1):
                beg, end = BIN_DUR * (pos - 1), BIN_DUR * pos
                bucket[pos - 1].append(_theta_power(tfr, beg, end))

    # per‑subject average across trials ---------------------------------------
    mem_row  = [np.nanmean(v) if v else np.nan for v in mem_by_pos]
    ctrl_row = [np.nanmean(v) if v else np.nan for v in ctrl_by_pos]

    mem_all.append(mem_row)
    ctrl_all.append(ctrl_row)

# ---------------------------------------------------------------------
# GRAND AVERAGE PLOT ---------------------------------------------------
# ---------------------------------------------------------------------
mem_all  = np.asarray(mem_all,  float)
ctrl_all = np.asarray(ctrl_all, float)

plt.figure(figsize=(9, 5))
plt.errorbar(X_POS, np.nanmean(mem_all,  axis=0),
             yerr=sem(mem_all,  axis=0, nan_policy='omit'),
             marker='o', label='Memory', linewidth=2)
plt.errorbar(X_POS, np.nanmean(ctrl_all, axis=0),
             yerr=sem(ctrl_all, axis=0, nan_policy='omit'),
             marker='s', label='Control', linewidth=2)
plt.xlabel('Serial position (digit)')
plt.ylabel('Theta power  (% change vs baseline)')
plt.title('Replication of Kosachenko et al. 2023 — Fig 5c (theta)')
plt.xticks(X_POS)
plt.ylim(bottom=0)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'theta_replication.png'), dpi=300)
plt.show()

# ==============================================================
#  OPTIONAL: inspect individual curves and flag outliers
# ==============================================================
import pathlib
from matplotlib.ticker import PercentFormatter

PLOT_INDIVIDUAL = True          # set False to skip the plots
OUT_DIR         = pathlib.Path('subject_curves_CSD')
OUT_DIR.mkdir(exist_ok=True)

suspects = []                   # will collect subject IDs > 40 %

for subj_idx, subj in enumerate(SUBJECTS):
    mem_curve  = mem_all[subj_idx]
    ctrl_curve = ctrl_all[subj_idx]

    if np.nanmax(np.abs(mem_curve)) > 0.40:   # 40 % threshold
        suspects.append(subj)

    if not PLOT_INDIVIDUAL:
        continue

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(X_POS, mem_curve,  '-o', label='Memory',  alpha=.8)
    ax.plot(X_POS, ctrl_curve, '-s', label='Control', alpha=.8)
    ax.set(title = f'Subject {subj:03d}',
           xlabel = 'Serial position (digit)',
           ylabel = 'Theta power (% vs baseline)')
    ax.xaxis.set_ticks(X_POS)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))  # 0.10 → 10 %
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f'sub-{subj:03d}_theta.png')
    plt.close(fig)

print('\nPossible outliers (|theta| > 40 %):', suspects)
