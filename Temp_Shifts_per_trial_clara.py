# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

subnums = np.arange(32, 98, 1)
subject_numbers = np.delete(subnums, np.where((subnums == 32) | (subnums == 37) |(subnums == 53) |(subnums == 61) | (subnums == 66) | (subnums == 78) | (subnums == 84) | (subnums == 94) | (subnums == 96)))
frontal_midline_channels = ['AFz', 'AF3', 'AF4', 'Fz', 'F1', 'F2', 'F3', 'F4', 'FC3', 'FC1', 'FC2', 'FC4', 'Cz', 'C3', 'C1', 'C2', 'C4']


# %%
def normalise_eeg(matrix: np.ndarray) -> np.ndarray:
    """
    matrix  shape: (n_samples, n_channels)
    1. centre each channel (remove DC offset)
    2. divide by a single scalar so that Var(all entries)=1
    """
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    # variance of the flattened array (ddof=0)
    var_total = np.mean(centred ** 2)
    return centred / np.sqrt(var_total)

# %%
from pathlib import Path
import pandas as pd
import numpy as np

PUPIL_ROOT = Path(r"C:\Users\cdd\Documents\Uni\Special_course\pupil_processed_clara")
EEG_ROOT   = Path(r"C:\Users\cdd\Documents\Uni\Special_course\code\Special_course\eeg_theta_processed2")
SHIFTS     = np.arange(-10, 11)          # ±1 s at 10 Hz

results = []   # will collect one row per (sub,cond,load,epoch,chan)

for sub_path in EEG_ROOT.iterdir():        # sub-032, sub-033, …
    if not (sub_path.name.startswith("sub-0") and sub_path.is_dir()):
        continue
    if sub_path.name == "sub-032_old":  # skip missing subject
        continue
    
    # Skip subjects that are not in subject_numbers
    try:
        sub_num = int(sub_path.name.split('-')[1])
        if sub_num not in subject_numbers:
            print(f"Skipping {sub_path.name} we do not want to analyse it.")
            continue
    except (IndexError, ValueError):
        # In case the format is not as expected
        pass
    
    print(f"Processing {sub_path.name}...")

    for cond_path in sub_path.iterdir():   # memory / control
        for load_path in cond_path.iterdir():  # 05 / 09 / 13
            eeg_epochs  = sorted(load_path.glob("trial_*.csv"))

            pupil_path = PUPIL_ROOT / sub_path.name / cond_path.name / load_path.name
            if not pupil_path.exists():
                print(f"Skipping {pupil_path} as it does not exist.")
                continue
            pupil_epochs = sorted(pupil_path.glob("trial_*.csv"))
            print(f"Found {len(eeg_epochs)} EEG epochs and {len(pupil_epochs)} pupil epochs.")
            
            common = set(e.name for e in eeg_epochs) & set(p.name for p in pupil_epochs)
            print(f"Common trials: {len(common)}")

            for fname in common:
                eeg_df = pd.read_csv(load_path / fname, comment='#', skip_blank_lines=True, index_col=0) #first column is time
                pupil_df    = pd.read_csv(pupil_path / fname, comment='#',names=['time', 'diameter_z'], index_col=0)
                pupil = pupil_df['diameter_z'].values
                eeg      = normalise_eeg(eeg_df.values)                    # (T,C)
                pupil_z  = (pupil - pupil.mean()) / pupil.std(ddof=0)      # z-score per trial

                T1 = len(pupil_z)
                T2 = len(eeg_df)
                if T1 <= 40:
                    print(f"Skipping {fname} due to insufficient pupil data length ({T1} samples).")
                    continue
                if abs(T1 - T2) > 30:
                    print(f"Skipping {fname} due to mismatch in lengths ({T1} vs {T2} samples).")
                    continue
                T = T2
                if T1 < T2: T = T1
                win = slice(20, T - 20)  # use samples 20 to T-20 (inclusive)
                for c, chan in enumerate(eeg_df.columns):
                    best_r, best_shift = -np.inf, np.nan
                    for s in SHIFTS:
                        eeg_shifted = np.roll(eeg[:, c], s)
                        eeg_shifted_window = eeg_shifted[win]  # only use the window of interest
                        #pupil_window
                        r = np.corrcoef(pupil_z[win], eeg_shifted_window)[0, 1]
                        if r > best_r:
                            best_r, best_shift = r, s*100   # samples→ms
                    if np.isfinite(best_r):
                        results.append({
                            "subject": sub_path.name,
                            "condition": cond_path.name,
                            "load": int(load_path.name),
                            "epoch": fname,
                            "channel": chan,
                            "r_max": best_r,
                            "lag_ms": best_shift
                        })

out = pd.DataFrame(results)
out.to_csv("trial_level_correlations.csv", index=False)

# %%
import pandas as pd
from pathlib import Path

FILE = Path("trial_level_correlations.csv")       # adjust if you saved it elsewhere
assert FILE.exists(), f"{FILE} not found – run the correlation script first!"

# 1)  load
df = pd.read_csv(FILE)                            # columns: subject, condition, load, epoch,
                                                  #          channel, r_max, lag_ms

# 2)  Fisher-z transform *optional* --------------
# If you prefer arithmetic means of the correlations themselves, skip this block.  
# Fisher-z makes the mean less biased when |r| is large.
use_fisher = False
if use_fisher:
    df["r_z"] = np.arctanh(df["r_max"])
    r_col = "r_z"
else:
    r_col = "r_max"

# 3)  aggregate: mean ± SD per condition & electrode
summary = (df
           .groupby(["condition", "channel"])
           .agg(r_mean   =(r_col,  "mean"),
                r_std    =(r_col,  "std"),
                lag_mean =("lag_ms","mean"),
                lag_std  =("lag_ms","std"))
           .reset_index())

# 4)  (re-)tanh if you used Fisher-z
if use_fisher:
    summary[["r_mean","r_std"]] = np.tanh(summary[["r_mean","r_std"]])

# 5)  nice formatting
summary = (summary
           .assign(r_mean =lambda d: d.r_mean.round(3),
                   r_std  =lambda d: d.r_std .round(3),
                   lag_mean =lambda d: d.lag_mean.astype(int),
                   lag_std  =lambda d: d.lag_std .round(1))
           .rename(columns={"r_mean":"corr µ",
                            "r_std":"corr σ",
                            "lag_mean":"lag µ (ms)",
                            "lag_std":"lag σ (ms)"}))

summary.to_csv("correlation_summary_by_condition.csv", index=False)   # optional export


# %%
"""
Files expected
--------------
trial_level_correlations.csv  # produced by the per-trial correlation script

Output
------
Loads the per-trial correlation file, computes a paired, one-tailed memory > control t-test for every electrode, 
corrects the p-values with Benjamini-Hochberg FDR

A DataFrame in the variable `res_df` plus an optional CSV on disk.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ------------------------------------------------------------------
CSV_PATH = Path("trial_level_correlations.csv")   # adjust if needed
ALPHA    = 0.05                                   # FDR threshold
SAVE_CSV = True                                   # set False to skip saving
# ------------------------------------------------------------------

# 1) load and keep finite correlations only
df = pd.read_csv(CSV_PATH)
df = df[np.isfinite(df["r_max"])]

# 2) mean correlation per SUBJECT × CONDITION × CHANNEL
#    → a wide table with two columns: 'control' and 'memory'
grid = (df.groupby(["subject", "condition", "channel"])["r_max"]
          .mean()
          .unstack(level="condition"))            # shape: (subj*chan) × 2

# 3) paired t-test (memory > control) electrode-wise
results = []
for ch in grid.index.get_level_values("channel").unique():
    sub_tbl = grid.xs(ch, level="channel", drop_level=False)

    # keep only subjects with BOTH conditions
    paired = sub_tbl.dropna(subset=["control", "memory"])
    if len(paired) < 3:          # need at least 3 paired subjects
        continue

    # two-sided paired test, then convert to one-tailed
    t_stat, p_two = stats.ttest_rel(paired["memory"], paired["control"])
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2

    results.append({
        "channel"   : ch,
        "n_subjects": len(paired),
        "mean_mem"  : paired["memory"].mean(),
        "mean_ctl"  : paired["control"].mean(),
        "mean_diff" : paired["memory"].mean() - paired["control"].mean(),
        "t_stat"    : t_stat,
        "p_one"     : p_one
    })

res_df = pd.DataFrame(results)

# 4) FDR correction across electrodes
if not res_df.empty:
    reject, p_fdr, _, _ = multipletests(res_df["p_one"],
                                        alpha=ALPHA,
                                        method="fdr_bh")
    res_df["p_FDR"]   = p_fdr
    res_df["sig_FDR"] = np.where(reject, "★", "")

# 5) tidy formatting
res_df = (res_df
          .sort_values("p_one")
          .round({"mean_mem": 3, "mean_ctl": 3,
                  "mean_diff": 3, "t_stat": 3,
                  "p_one": 4, "p_FDR": 4}))

print(res_df.to_string(index=False))

if SAVE_CSV:
    res_df.to_csv("memory_vs_control_stats_by_electrode.csv", index=False)
    print("\nSaved → memory_vs_control_stats_by_electrode.csv")


# %%
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
'''Are correlations larger in the memory condition than in control, ignoring which electrode they come from?”

the cleanest approach is:

Reduce each subject to one number per condition
Average the correlations across all electrodes (or across electrodes and trials) so every participant contributes exactly two scores: one for memory, one for control.

Run a paired test across subjects
A paired, one-tailed t-test (memory > control) is the simplest.
If you prefer to keep every electrode as a repeated measure you can switch to a linear-mixed model, but the conclusion is usually identical.'''
# -------- settings --------
CSV_PATH = Path("trial_level_correlations.csv")   # same file as before
MIN_TRIALS = 10     # ignore subject × condition cells with fewer trials
# --------------------------

# 1) load & drop infinities / NaNs
df = pd.read_csv(CSV_PATH)
df = df[np.isfinite(df["r_max"])]

# 2) mean across *all* trials and electrodes for every subject × condition
sub_cond = (df.groupby(["subject", "condition"])["r_max"]
              .agg(["mean", "count"])
              .rename(columns={"mean":"r_mean", "count":"n_trials"}))

# 3) keep only cells with enough trials
sub_cond = sub_cond[sub_cond["n_trials"] >= MIN_TRIALS]

# 4) reshape so each row = one subject with both conditions
wide = (sub_cond.reset_index()
                   .pivot(index="subject", columns="condition", values="r_mean")
                   .dropna())      # drops subjects missing either condition

print(f"Paired subjects kept: {len(wide)}")

# 5) paired, one-tailed t-test (memory > control)
t_stat, p_two = stats.ttest_rel(wide["memory"], wide["control"])
p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2

print(f"\nMean r  (memory):  {wide['memory'].mean():.3f}")
print(f"Mean r  (control): {wide['control'].mean():.3f}")
print(f"Mean Δr (mem-ctl): { (wide['memory'] - wide['control']).mean():.3f}")
print(f"\nPaired t-stat   : {t_stat:.3f}")
print(f"One-tailed p     : {p_one:.4f}")


# %%
'''
tests, within the memory condition only, whether the average correlation differs 
between the 5-, 9-, and 13-digit loads
'''
# ---------------------------------------------------------------
#  STEP 0 – load the per-trial correlations (if not already in `out`)
# ---------------------------------------------------------------
import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

out = pd.read_csv("trial_level_correlations.csv")      # skip if `out` is still in RAM

# ---------------------------------------------------------------
#  STEP 1 – keep MEMORY trials, compute a mean r per subject × load
# ---------------------------------------------------------------
mem = out[out["condition"] == "memory"]

sub_load = (mem.groupby(["subject", "load"])["r_max"]
              .mean()                       # mean across all trials & channels
              .unstack())                   # columns 5, 9, 13

#  keep only subjects that have *all three* loads
sub_load = sub_load.dropna()

print(f"Subjects with all loads: {len(sub_load)}")

# ---------------------------------------------------------------
#  STEP 2 – paired comparisons 5 vs 9, 5 vs 13, 9 vs 13
# ---------------------------------------------------------------
alpha = 0.05
pairs = [(5, 9), (5, 13), (9, 13)]
records = []

for a, b in pairs:
    diff = sub_load[a] - sub_load[b]        # paired difference for each subject
    t_stat, p_two = stats.ttest_rel(sub_load[a], sub_load[b])
    mean_diff = diff.mean()

    # one-tailed p: “is load a > load b ?”
    p_one = p_two / 2 if mean_diff > 0 else 1 - p_two / 2

    records.append(dict(pair=f"{a} vs {b}",
                        mean_a=sub_load[a].mean(),
                        mean_b=sub_load[b].mean(),
                        mean_diff=mean_diff,
                        t_stat=t_stat,
                        p_one=p_one))

pair_df = pd.DataFrame(records)

# ---------------------------------------------------------------
#  STEP 3 – FDR-correct across the three comparisons
# ---------------------------------------------------------------
rej, p_fdr, _, _ = multipletests(pair_df["p_one"],
                                 alpha=alpha,
                                 method="fdr_bh")
pair_df["p_FDR"]   = p_fdr
pair_df["sig_FDR"] = np.where(rej, "★", "")

# tidy formatting
pair_df = pair_df.round({"mean_a":3, "mean_b":3, "mean_diff":3,
                         "t_stat":3, "p_one":4, "p_FDR":4})

print("\nMemory-condition load comparison (one-tailed, FDR-corrected)")
print(pair_df.to_string(index=False))



