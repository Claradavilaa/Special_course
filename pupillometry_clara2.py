# %% [markdown]
# Same as the other but we save the epochs
# 
# Save trial‑level, baseline‑corrected and z‑scored pupil epochs for each subject.
# 
# The script replicates the preprocessing pipeline implemented in `pupillometry_clara.py`
# (and keeps all parameter choices), but adds automatic export of **every single epoch**
# into a project‑wide folder structure that looks like this:
# 
# pupil_processed_clara/
# ├── sub-032/
# │   ├── memory/05/epoch_000.csv
# │   ├── memory/05/epoch_001.csv
# │   ├── ...
# │   ├── memory/09/epoch_000.csv
# │   ├── control/13/epoch_028.csv
# │   └── ...
# ├── sub-033/
# │   └── ...
# ├── …
# ├── subject_skip_log.tsv                    # subjects skipped with reason
# └── trial_skip_log.tsv                      # each skipped trial with reason
# 
# *   **No averaging** is performed – every trial is stored on its own.
# *   Each CSV contains a single column `diameter` (10 Hz, in seconds from −2 s).
#     Additional metadata are written as header comments for convenience.
# *   Subjects with < 6 valid trials are skipped (same rule as in the original script).
# 

# %%
from __future__ import annotations
import os
import pathlib
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# -----------------------------------------------------------------------------
#  USER CONFIGURATION
# -----------------------------------------------------------------------------
DATASET_ROOT = pathlib.Path("C:/Users/cdd/Documents/Uni/Special_course/ds003838-download")
OUTPUT_ROOT  = pathlib.Path("C:/Users/cdd/Documents/Uni/Special_course/pupil_processed_clara")
SUBJECTS     = np.setdiff1d(np.arange(32, 99), [37, 66, 94])
# -----------------------------------------------------------------------------

OUTPUT_ROOT.mkdir(exist_ok=True)
SUBJECT_SKIP_FILE = OUTPUT_ROOT / "subject_skip_log.tsv"
TRIAL_SKIP_FILE   = OUTPUT_ROOT / "trial_skip_log.tsv"

subject_skip_records: List[dict] = []
trial_skip_records:   List[dict] = []

# %%
# -----------------------------------------------------------------------------
#  Helper functions (copied / adapted from pupillometry_clara.py)
# -----------------------------------------------------------------------------

def group_trials(events_df: pd.DataFrame) -> List[Tuple[str, int, pd.DataFrame]]:
    """Split `events.tsv` rows into individual trials.

    Returns a list of tuples: (condition, load, rows)
    condition - "memory" / "control"; load - 5 / 9 / 13.
    """
    events_df = events_df.copy()
    events_df["label"] = events_df["label"].astype(str)

    # boolean mask for where new trials start (labels with digit 01)
    new_trial_mask = events_df["label"].str[2:4] == "01" 

    # Assign trial numbers using cumulative sum of new trial indicators
    events_df["trial_id"] = new_trial_mask.cumsum()

    # Extract metadata: condition and load
    events_df["condition"] = np.where(events_df["label"].str.startswith("50"), "control", "memory")
    events_df["load"] = events_df["label"].str[4:6].astype(int)

    grouped: List[Tuple[str, int, pd.DataFrame]] = []
    for _, g in events_df.groupby("trial_id"):
        grouped.append((g["condition"].iloc[0], g["load"].iloc[0], g.drop(columns="trial_id")))
    return grouped


def apply_speed_mad_filter(pupil_epoch: pd.DataFrame, time_col="pupil_timestamp", signal_col="diameter", factor=16):
    """Invalidate samples where 1st-derivative (speed) is an extreme outlier."""
    # Compute time difference (in seconds)
    dt = np.diff(pupil_epoch[time_col], prepend=pupil_epoch[time_col].iloc[0])
    dt[dt <= 0] = np.nan # Remove zero or negative dt to prevent spikes

    # Compute dilation speed: absolute change in diameter / time difference
    speed = np.abs(np.diff(pupil_epoch[signal_col], prepend=pupil_epoch[signal_col].iloc[0])) / dt

    # Mask speed where either signal or time was NaN
    nan_mask = pupil_epoch[signal_col].isna() | pd.isna(dt)
    speed[nan_mask] = np.nan
    
    # Compute MAD and median
    median_speed = np.nanmedian(speed)
    mad_speed = np.nanmedian(np.abs(speed - median_speed))

    # Threshold
    threshold = median_speed + factor * mad_speed

    # Invalidate points with too high speed
    pupil_epoch.loc[speed > threshold, signal_col] = np.nan
    return pupil_epoch


def preprocess_epoch(epoch: pd.DataFrame) -> pd.Series | None:
    """Apply the full preprocessing chain to one epoch.

    Returns the *processed* diameter Series (indexed by time) or `None` if the
    epoch should be rejected (too many NaNs).
    """
    # 1) speed‑based artefact rejection (MAD 16×)
    epoch = apply_speed_mad_filter(epoch)

    # 2) Handle blink: Set 'diameter' to NaN where blinks occur (±100ms)
    if (epoch["blink"] == 1).any():
        blink_times = epoch.loc[epoch["blink"] == 1, "pupil_timestamp"].values
        mask = np.zeros(len(epoch), dtype=bool)
        for ts in blink_times:
            mask |= (epoch["pupil_timestamp"] >= ts - 0.1) & (epoch["pupil_timestamp"] <= ts + 0.1)
        epoch.loc[mask, "diameter"] = np.nan

    # 3) reject if > 50 % missing
    if epoch["diameter"].isna().mean() > 0.5: 
        # TODO: I should somehow markt these trials because I will need to remove them from the EEG analysis too, since i am doing CCA per subject and trial
        # Same with removed subjects
        return None

    # 4) interpolate, 5‑point moving average
    epoch["diameter"] = (epoch["diameter"].interpolate("linear", limit_direction="both")
                        .rolling(window=5, center=True, min_periods=1).mean())

    # convert to datetime index for resampling
    epoch["pupil_datetime"] = pd.to_datetime(epoch["pupil_timestamp"], unit="s")
    epoch = epoch.set_index("pupil_datetime")

    # 5) baseline (−2 s … 0 s relative to first digit)
    baseline = epoch.loc[epoch.index < epoch.index[0] + pd.Timedelta(seconds=2), "diameter"].mean()
    epoch["diameter"] = epoch["diameter"] - baseline

    # 6) resample to 100 Hz (10 ms) and z‑score **per trial**
    epoch = epoch.resample("10ms").mean()
    epoch["diameter"] = epoch["diameter"].ffill().bfill()
    #epoch["diameter_z"] = (epoch["diameter"] - epoch["diameter"].mean()) / epoch["diameter"].std(ddof=0)

    # 8) replace index with numeric time vector (s,  –2 … +len)
    epoch["time"] = ((epoch.index - epoch.index[0]).total_seconds() - 2).round(2)
    epoch = epoch.set_index("time")

    return epoch["diameter"]

# %%
# -----------------------------------------------------------------------------
#  Main loop
# -----------------------------------------------------------------------------

def process_subject(sub_id: int):
    sub_tag = f"sub-{sub_id:03d}"
    print(f"▶  {sub_tag}")

    # load events
    ev_path = DATASET_ROOT / sub_tag / "pupil" / f"{sub_tag}_task-memory_events.tsv"
    pupil_path = DATASET_ROOT / sub_tag / "pupil" / f"{sub_tag}_task-memory_pupil.tsv"
    
    # save directory
    subj_root = OUTPUT_ROOT / sub_tag
    if subj_root.exists():
        print(f"Output directory {subj_root} already exists, skipping")
        return

    if not ev_path.exists() or not pupil_path.exists():
        subject_skip_records.append({"subject": sub_tag, "reason": "missing_files"})
        print("   missing files - skipped")
        return

    events = pd.read_csv(ev_path, sep="\t", usecols=["timestamp", "label"])
    pupil = pd.read_csv(pupil_path, sep="\t", usecols=["pupil_timestamp", "diameter", "confidence", "blink"])

    # invalidate low‑confidence samples
    pupil.loc[pupil["confidence"] < 0.8, "diameter"] = np.nan

    # ------------------------------------------------------------------
    epochs_meta = group_trials(events)
    processed_epochs: Dict[Tuple[str, int], List[pd.Series]] = {}

    for trial_idx, (cond, load, rows) in enumerate(epochs_meta):
        start = rows["timestamp"].iloc[0] - 2 # 2s before first digit
        end = rows["timestamp"].iloc[-1] + 3 # 3s after last digi
        epoch_raw = pupil[(pupil["pupil_timestamp"] >= start) & (pupil["pupil_timestamp"] <= end)].copy()
        if epoch_raw.empty:
            trial_skip_records.append({"subject": sub_tag, "condition": cond, "load": load, "trial_idx": trial_idx, "reason": "no_samples"})
            continue

        processed = preprocess_epoch(epoch_raw)
        if processed is None:
            trial_skip_records.append({"subject": sub_tag, "condition": cond, "load": load, "trial_idx": trial_idx, "reason": ">50%_nan"})
            continue

        processed_epochs.setdefault((cond, load), []).append(processed)

    n_valid = sum(len(v) for v in processed_epochs.values())
    print(f"   kept {n_valid} valid epochs")
    
    if n_valid < 6:
        subject_skip_records.append({"subject": sub_tag, "reason": "<6_valid_epochs"})
        return

    # ------------------------------------------------------------------
    #  SAVE
    # ------------------------------------------------------------------

    for (cond, load), epochs in processed_epochs.items():
        target_dir = subj_root / cond / f"{load:02d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for idx, ep in enumerate(epochs):
            out_file = target_dir / f"epoch_{idx:03d}.csv"
            header_comment = (
                f"# subject={sub_tag} condition={cond} load={load} sample_rate=100Hz\n"
                f"# columns: time(s), diameter\n"
            )
            ep.to_csv(out_file, header=False, index=True, float_format="%.6f", date_format="%.1f")
            # prepend metadata comment
            with open(out_file, "r+") as f:
                content = f.read()
                f.seek(0, 0)
                f.write(header_comment + content)


# %%

for sid in SUBJECTS:
    process_subject(int(sid))

# write logs – overwrite on each run for reproducibility
pd.DataFrame(subject_skip_records).to_csv(SUBJECT_SKIP_FILE, sep="\t", index=False)
pd.DataFrame(trial_skip_records  ).to_csv(TRIAL_SKIP_FILE,   sep="\t", index=False)

print("\n✅  Finished. Logs written to:")
print("   •", SUBJECT_SKIP_FILE)
print("   •", TRIAL_SKIP_FILE)




