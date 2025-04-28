# -*- coding: utf-8 -*-
"""EEG group θ-power (debug, 2 subjects, verbose timing)"""

from __future__ import annotations
import time, sys
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ CONFIG ----
PATH_DATA = Path(r"C:/Users/cdd/Documents/Uni/Special_course/ds003838-download")
PATH_OUT  = Path("Theta_Processed_Group_DEBUG")
PATH_OUT.mkdir(exist_ok=True)

SUBJECTS   = np.setdiff1d(np.arange(32, 99), [37, 66, 94])
CH_FRONTAL = ["AFz","AF3","AF4","Fz","F1","F2","F3","F4",
              "FC3","FC1","FC2","FC4","Cz","C3","C1","C2","C4"]

DECIM      = 10                  # 1 kHz → 100 Hz
DT         = DECIM / 1000        # sec / sample (0.01 s)
TMIN       = -2.0
TMAX       = {"05":11.0, "09":17.0, "13":27.0}
BASELINE   = (-1.0, 0.0)

EVENTS_MEM  = {"05":["6001050","6001051"],
               "09":["6001090","6001091"],
               "13":["6001130","6001131"]}
EVENTS_CTL  = {"05":["500105"],
               "09":["500109"],
               "13":["500113"]}

# ----------------------------------------------------------- small helpers ----
def tic(msg:str):
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)

def load_raw(sub:int):
    f = PATH_DATA/f"sub-{sub:03d}"/"eeg"/f"sub-{sub:03d}_task-memory_eeg.set"
    tic(f"[{sub:03d}] read_raw")
    raw = mne.io.read_raw_eeglab(f, preload=True, verbose="ERROR")
    tic(f"[{sub:03d}]  …done ({raw.n_times/raw.info['sfreq']:.1f} s raw)")
    tic(f"[{sub:03d}] resample→100 Hz")
    raw.resample(100, npad="auto", verbose="ERROR")
    tic(f"[{sub:03d}]  …done")
    tic(f"[{sub:03d}] θ-filter (IIR)")
    raw.filter(4, 8, picks="eeg",
               method="iir", iir_params=dict(order=4, ftype="butter"),
               verbose="ERROR")
    tic(f"[{sub:03d}]  …done")
    raw.set_eeg_reference("average", verbose="ERROR")
    return raw

def make_epochs(raw, load:str, cond:str):
    codes = EVENTS_MEM[load] if cond=="memory" else EVENTS_CTL[load]
    events, ids = mne.events_from_annotations(raw, verbose="ERROR")
    mapping = {c: ids[c] for c in codes if c in ids}
    if not mapping:
        return None
    return mne.Epochs(raw, events, mapping, tmin=TMIN, tmax=TMAX[load],
                      picks="eeg", preload=True, baseline=None, verbose="ERROR")

def theta_env(epochs:mne.Epochs):
    if epochs is None or len(epochs)==0:
        return np.empty((0,0))
    epochs = epochs.copy().pick(CH_FRONTAL)
    epochs.apply_hilbert(envelope=True, verbose="ERROR")
    data = epochs.get_data()**2                    # power
    b0, b1 = int((BASELINE[0]-TMIN)/DT), int((BASELINE[1]-TMIN)/DT)
    base   = data[:,:,b0:b1].mean(-1, keepdims=True)
    pc     = (data-base)/base*100.                 # % change
    return pc.mean(1)                              # (n_trials, n_times)

def subject_dict(sub:int):
    try:
        raw = load_raw(sub)
    except Exception as e:
        print(f"⚠ sub-{sub:03d}: {e}")
        return {}
    out={}
    for load in ("05","09","13"):
        for cond in ("memory","control"):
            tic(f"[{sub:03d}] epochs {cond} {load}")
            ep = make_epochs(raw, load, cond)
            env= theta_env(ep)
            if env.size:
                out[(cond,load)] = env.mean(0)     # average over trials
            tic(f"[{sub:03d}]  …{len(env)} trials")
    return out

def grand_average(subj_list):
    grand={}
    for key in [(c,l) for c in("memory","control") for l in("05","09","13")]:
        series=[d[key] for d in subj_list if key in d]
        if not series: continue
        mlen=min(map(len, series))
        arr = np.stack([s[:mlen] for s in series])
        grand[key]=(arr.mean(0), arr.std(0,ddof=1)/np.sqrt(arr.shape[0]))
    return grand

# --------------------------------------------------------------- main ----
tic("=== start ===")
subs_data=[d for s in SUBJECTS if (d:=subject_dict(s))]
tic("subjects processed")

if not subs_data:
    sys.exit("No usable data!")

grand = grand_average(subs_data)
pts   = max(len(v[0]) for v in grand.values())
time_vec = np.arange(pts)*DT + TMIN

# save CSVs + quick plot
for (cond,load),(m,s) in grand.items():
    pd.DataFrame({"time":time_vec[:len(m)],"mean":m,"sem":s}
        ).to_csv(PATH_OUT/f"theta_{cond}_{load}.csv", index=False)

# -----------------------------------------------------------------
# 4. PLOT  – two panels: Memory | Control
# -----------------------------------------------------------------
t_axis = np.arange(pts) * DT + TMIN           # avoid clashing with time module

fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
cond_colors = {"05": "tab:blue", "09": "tab:green", "13": "tab:red"}

for ax, cond in zip(axs, ["memory", "control"]):
    for load, color in cond_colors.items():
        if (cond, load) not in grand:
            continue
        mean, sem = grand[(cond, load)]
        ax.plot(t_axis[: len(mean)], mean, label=f"Load {load}", color=color)
        ax.fill_between(
            t_axis[: len(mean)],
            mean - sem,
            mean + sem,
            alpha=0.25,
            color=color,
        )

    ax.set_title(cond.capitalize())
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.grid(alpha=0.3)

axs[0].set_ylabel("θ Δ% (frontal)")
fig.suptitle("Grand-average frontal θ-power")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.legend(frameon=False, loc="upper right")
plt.show()


tic("✓ Done")
