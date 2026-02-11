from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr 
from typing import List, Tuple
import json

###############################################################################
# Paths & constants
###############################################################################
WIN_OFFSET1 = 200                  # discard first 2 s
WIN_OFFSET2 = 101                  # discard last 1.01 s

###############################################################################
# Helper functions
###############################################################################

def normalise_eeg(x: np.ndarray) -> np.ndarray:
    """Centre each channel and scale so Σ x² = 1 over time×channels."""
    x = x - x.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(x**2))
    return x / scale


def cca_corr(eeg: np.ndarray, pupil: np.ndarray) -> float:
    """Canonical correlation (single component)."""
    cca = CCA(n_components=1, max_iter=1000, scale=False, tol=1e-6)
    cca.fit(eeg, pupil)
    u, v = cca.transform(eeg, pupil)
    return float(np.corrcoef(u[:, 0], v[:, 0])[0, 1])

def candidate_lags_units(step_ms=100, max_ms=2000):
    # Your code uses 10 ms units (lag_ms = shift*10)
    step_units = step_ms // 10
    max_units  = max_ms  // 10
    return list(range(-max_units, max_units + 1, step_units))

def load_all_trials(
        sub: int,
        eeg_root: Path,
        pupil_root: Path,
        min_len: int = 40,
        max_len_diff: int = 30,
) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    """
    Load *all* valid EEG-pupil trial pairs for one subject.

    Parameters
    ----------
    sub : int
        Numeric subject ID (e.g. 42).
    eeg_root, pupil_root : Path
        Roots of the pre-processed EEG and pupil folders.
    min_len : int
        Minimum number of samples a pupil trace must have to be accepted.
    max_len_diff : int
        Reject trial if |len(pupil)-len(eeg)| exceeds this.
    Returns
    -------
    trials : list of (eeg, pupil_z, meta)
        * eeg        - (T × n_channels) float64, already centred/scaled
        * pupil_z    - (T × 1) float64, per-trial z-scored
        * meta       - dict with subject/condition/load/epoch
    """
    trials = []
    sub_tag = f"sub-{sub:03d}"
    eeg_sub  = eeg_root   / sub_tag
    pupil_sub = pupil_root / sub_tag

    if not eeg_sub.exists():
        print(f"{sub_tag}: EEG folder missing - skipped")
        return trials

    # iterate condition (“control” / “memory”) and load (“05” / “09” / “13”)
    for cond_path in sorted(eeg_sub.iterdir()):
        if not cond_path.is_dir():
            continue
        for load_path in sorted(cond_path.iterdir()):
            if not load_path.is_dir():
                continue

            # matching pupil directory
            pupil_path = pupil_sub / cond_path.name / load_path.name
            if not pupil_path.exists():
                continue

            eeg_epochs   = sorted(load_path.glob("trial_*.csv"))
            pupil_epochs = sorted(pupil_path.glob("trial_*.csv"))
            common = {f.name for f in eeg_epochs} & {f.name for f in pupil_epochs}
            if not common:
                continue

            for fname in sorted(common):
                eeg_df = pd.read_csv(load_path / fname, comment="#", index_col=0)
                pupil_df = pd.read_csv(pupil_path / fname, comment="#", skiprows=1,
                                       names=["time", "diameter_z"], index_col=0)

                eeg   = eeg_df.values.astype(float)
                pupil = pupil_df["diameter_z"].values.astype(float)

                # basic validity checks
                if len(pupil) < min_len or abs(len(pupil) - len(eeg)) > max_len_diff:
                    continue

                # normalise signals ---------------------------------------- now do it after trimming shifted trials
                #eeg_norm = normalise_eeg(eeg)           # your helper from before
                #pupil_z  = ((pupil - pupil.mean()) / pupil.std(ddof=0))

                # same number of samples
                T = min(len(eeg), len(pupil))
                eeg = eeg[0:T, :]  # (T × n_channels)
                pupil  = pupil[0:T].reshape(-1, 1)

                meta = {
                    "subject":   sub_tag,
                    "condition": cond_path.name,
                    "load":      int(load_path.name),
                    "epoch":     fname
                }
                trials.append((eeg, pupil, meta))

    return trials

def split_trials_by_condition(
        trials: List[Tuple[np.ndarray, np.ndarray, dict]],
        memory_label: str = "memory",
        control_label: str = "control"
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, dict]], List[Tuple[np.ndarray, np.ndarray, dict]]]:
    """
    Separate a mixed list of (eeg, pupil, meta) trial tuples into memory-condition and control-condition sub-lists.

    Parameters
    ----------
    trials : list of tuples
        Each tuple = (eeg_array, pupil_array, meta_dict).
        meta_dict must contain a key 'condition'.
    memory_label : str
        The value of meta['condition'] that marks a memory trial.
    control_label : str
        The value of meta['condition'] that marks a control trial.

    Returns
    -------
    memory_trials  : list[tuple]
    control_trials : list[tuple]
    """
    memory_trials  = []
    control_trials = []

    for eeg, pupil, meta in trials:
        cond = meta.get("condition", "").lower()
        if cond == memory_label:
            memory_trials.append((eeg, pupil, meta))
        elif cond == control_label:
            control_trials.append((eeg, pupil, meta))
        else: raise ValueError(f"Unknown condition label: {cond}")

    return memory_trials, control_trials

def split_trials_by_load(trials: List[Tuple[np.ndarray, np.ndarray, dict]]
                         ) -> Tuple[List[Tuple[np.ndarray, np.ndarray, dict]], List[Tuple[np.ndarray, np.ndarray, dict]], List[Tuple[np.ndarray, np.ndarray, dict]]]:
    """
    Separate a mixed list of (eeg, pupil, meta) trial tuples into 05, 09 and 13 load sub-lists.

    Parameters
    ----------
    trials : list of tuples
        Each tuple = (eeg_array, pupil_array, meta_dict).
        meta_dict must contain a key 'load'.

    Returns
    -------
    load_05_trials  : list[tuple]
    load_09_trials  : list[tuple]
    load_13_trials  : list[tuple]
    """
    load_05_trials = []
    load_09_trials = []
    load_13_trials = []

    for eeg, pupil, meta in trials:
        load = meta.get("load")
        if load == 5:
            load_05_trials.append((eeg, pupil, meta))
        elif load == 9:
            load_09_trials.append((eeg, pupil, meta))
        elif load == 13:
            load_13_trials.append((eeg, pupil, meta))
        else: raise ValueError(f"Unknown load label: {load}")

    return load_05_trials, load_09_trials, load_13_trials

# trials  : list of (eeg, pupil_z, meta)   -- the tuples returned by load_all_trials
# shift   : integer sample shift (best_shift)
# win     : slice or None                  -- cropping window (set to None if the
#                                            trials are already pre-trimmed)
def concat_trials(
    trials: List[Tuple[np.ndarray, np.ndarray, dict]],
    shift: int = 0,  # samples; positive = EEG delayed (EEG after pupil)
    trim=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate trials for CCA without circular wrap.

    Returns
    -------
    X : ndarray, shape (Σ T_eff, n_channels)   # EEG
    Y : ndarray, shape (Σ T_eff, 1)            # pupil
    """
    X_blocks, Y_blocks = [], []
    if trim is not None:
        target_trim_samp = trim // 10   # 5000 ms → 500 samples at 100 Hz
        extra_samp = max(0, (target_trim_samp - abs(shift)) // 2)
        #print(f"Lag is {shift} samples. Trimming {extra_samp} samples from each end after lag adjustment")

    for eeg, pupil_z, _ in trials:
        # Ensure 1D/2D shapes: EEG (T, C), pupil (T, 1)
        if eeg.ndim == 1:
            eeg = eeg[:, None]
        if pupil_z.ndim == 1:
            pupil_z = pupil_z[:, None]

        T = min(len(eeg), len(pupil_z))
        if T == 0:
            print("Warning: zero-length trial encountered - skipped")
            continue

        # Optional common window first (keeps boundaries away)
        start = WIN_OFFSET1
        stop  = T - WIN_OFFSET2
        if stop <= start:
            print("Warning: collapsed window after trimming - skipped")
            continue  # window collapsed; skip this trial

        eeg_w   = eeg[start:stop]
        pupil_w = pupil_z[start:stop]
        Tw = len(eeg_w)

        # Apply lag by asymmetric trimming (no roll, no wrap)
        # Positive shift => EEG delayed => drop 'shift' from EEG start and pupil end
        # Negative shift => EEG advanced => drop '-shift' from pupil start and EEG end
        if shift >= 0:
            if Tw <= shift: continue  # nothing left after lag
            eeg_seg   = eeg_w[shift:]             # drop early EEG
            pupil_seg = pupil_w[:Tw - shift]      # drop late pupil
        else:
            s = -shift
            if Tw <= s: continue
            eeg_seg   = eeg_w[:Tw - s]            # drop late EEG
            pupil_seg = pupil_w[s:]               # drop early pupil

        if trim is not None:
            Lcur = int(len(eeg_seg))
            if extra_samp > 0:
                extra_eff = int(min(extra_samp, max(0, (Lcur - 2) // 2)))
                if extra_eff > 0:
                    eeg_seg   = eeg_seg[extra_eff : Lcur - extra_eff]
                    pupil_seg = pupil_seg[extra_eff : Lcur - extra_eff]

        eeg_seg = normalise_eeg(eeg_seg)           # your helper from before
        pupil_seg = ((pupil_seg - pupil_seg.mean()) / pupil_seg.std(ddof=0))

        # Shapes now match
        X_blocks.append(eeg_seg)
        Y_blocks.append(pupil_seg)

    if not X_blocks:
        print("Warning: no valid trials after concatenation - returning empty arrays")
        return np.empty((0, trials[0][0].shape[-1])), np.empty((0, 1))

    X = np.vstack(X_blocks)
    Y = np.vstack(Y_blocks)
    return X, Y

def iterate_trials(trials, shift):
    """
    Yield (eeg_aligned, pupil_aligned, meta) one by one, with no circular wrap.

    Conventions
    -----------
    - `shift` in samples, along time axis (axis=0).
    - shift >= 0  → EEG delayed  → drop early EEG, drop late pupil
    - shift <  0  → EEG advanced → drop late EEG, drop early pupil
    """
    for eeg, pupil_z, meta in trials:
        # Window first
        T = min(len(eeg), len(pupil_z))
        start = WIN_OFFSET1
        stop  = T - 50
        if stop <= start:
            print("Warning: collapsed window after trimming - skipped")
            continue  # window collapsed; skip this trial

        eeg_w   = eeg[start:stop]
        pupil_w = pupil_z[start:stop]

        # Ensure both have time along axis 0 and share the same window length
        Tw = len(eeg_w)
        
        if Tw == 0: continue

        if shift >= 0: # EEG delayed: drop early EEG, drop late pupil
            eeg_out   = eeg_w[shift:]           # works for 1D or 2D (time on axis 0)
            pupil_out = pupil_w[:Tw - shift]
        else:
            s = -shift # EEG advanced: drop late EEG, drop early pupil
            
            eeg_out   = eeg_w[:Tw - s]
            pupil_out = pupil_w[s:]

        eeg_out = normalise_eeg(eeg_out)           # your helper from before
        pupil_out = ((pupil_out - pupil_out.mean()) / pupil_out.std(ddof=0))

        yield eeg_out, pupil_out, meta.copy()


def save_cca_weights(cca, subject_tag, lag_ms, eeg_ch_names, condition, out_dir=Path("weights")):
    """
    Dump EEG & pupil canonical weights to CSV/JSON for one subject.

    Parameters
    ----------
    cca            : fitted sklearn.cross_decomposition.CCA
    subject_tag    : "sub-042"
    lag_ms         : e.g. -90
    eeg_ch_names   : list[str] same order as columns in your trial matrices
    out_dir        : destination folder (created if missing)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------- Pupil weight -> JSON ---------------------------------------
    w_pupil_og = float(cca.y_weights_[0, 0])   # scalar in 1-dim pupil case
    w_pupil = 1.0
    with open(out_dir / f"{subject_tag}_pupil_weight_{condition}_{lag_ms:+d}ms.json", "w") as fh:
        json.dump({"weight": w_pupil}, fh, indent=2)
    
    # ------- EEG weights -> tidy CSV ------------------------------------
    w_eeg = cca.x_weights_[:, 0] / w_pupil_og
    w_eeg = pd.Series(w_eeg, index=eeg_ch_names, name="weight")
    w_eeg.index.name = "channel"
    w_eeg.to_csv(out_dir / f"{subject_tag}_eeg_weights_{condition}_{lag_ms:+d}ms.csv")

    print(f"saved weights for {subject_tag} (lag {lag_ms:+d} ms)")

# Type alias just for readability
TrialTuple = Tuple[np.ndarray, np.ndarray, dict]

def make_triplets_by_load(
    trials_cond: List[TrialTuple],
    condition_name: str = "memory",
    rng: np.random.Generator | None = None,
    verbose: bool = True,
) -> List[List[TrialTuple]]:
    """
    Build triplets from trials of a *single condition* (all 'memory' or all 'control').

    Strategy:
    1. Separate trials into load buckets (5, 9, 13).
    2. Create as many 'balanced' triplets as possible with exactly 1×5, 1×9, 1×13.
    3. Collect all leftover trials, shuffle them, form additional triplets of size 3.
    4. If leftover count mod 3 != 0, attach the last 1–2 trials to existing triplets.

    Returns
    -------
    triplets : list of lists
        Each inner list is a group of trials (size 3, occasionally 4–5 at the end).
    """
    if rng is None:
        rng = np.random.default_rng(7)

    if len(trials_cond) == 0:
        if verbose:
            print(f"[{condition_name}] No trials – returning empty triplet list.")
        return []

    # 1) Split by load: we reuse your helper
    load_05, load_09, load_13 = split_trials_by_load(trials_cond)
    load_05 = list(load_05)
    load_09 = list(load_09)
    load_13 = list(load_13)

    # Shuffle within each load bucket for randomness
    rng.shuffle(load_05)
    rng.shuffle(load_09)
    rng.shuffle(load_13)

    n05, n09, n13 = len(load_05), len(load_09), len(load_13)
    if verbose:
        print(f"[{condition_name}] trials per load: 5→{n05}, 9→{n09}, 13→{n13}")

    total = len(trials_cond)
    if total < 3:
        if verbose:
            print(f"[{condition_name}] < 3 trials total; returning one 'triplet' with all trials.")
        return [trials_cond[:] ]

    # 2) Balanced triplets: one of each load 5,9,13
    n_balanced = min(n05, n09, n13)
    triplets: List[List[TrialTuple]] = []

    for i in range(n_balanced):
        triplets.append([
            load_05[i],
            load_09[i],
            load_13[i],
        ])

    if verbose and n_balanced > 0:
        print(f"[{condition_name}] Created {n_balanced} balanced 5–9–13 triplets.")

    # 3) Collect leftovers after using balanced part
    leftovers: List[TrialTuple] = []
    leftovers.extend(load_05[n_balanced:])
    leftovers.extend(load_09[n_balanced:])
    leftovers.extend(load_13[n_balanced:])

    # Sanity: also catch any weird loads if present
    # (in case some trials had load != 5/9/13)
    for tr in trials_cond:
        load = tr[2].get("load", None)
        if load not in (5, 9, 13):
            leftovers.append(tr)

    # Remove duplicates (in case weird loads were also in 05/09/13 lists)
    if leftovers:
        # Use object id to deduplicate while keeping order
        seen = set()
        unique_leftovers = []
        for tr in leftovers:
            oid = id(tr)
            if oid not in seen:
                seen.add(oid)
                unique_leftovers.append(tr)
        leftovers = unique_leftovers

    rng.shuffle(leftovers)
    n_left = len(leftovers)

    if verbose and n_left > 0:
        print(f"[{condition_name}] {n_left} leftover trials after balanced triplets.")

    # 4) Make extra triplets purely from leftovers (any load mix)
    q, r = divmod(n_left, 3)
    idx = 0
    for _ in range(q):
        triplets.append([
            leftovers[idx],
            leftovers[idx + 1],
            leftovers[idx + 2],
        ])
        idx += 3

    remaining = leftovers[idx:]  # length r ∈ {0, 1, 2}

    # 5) Attach remaining 1–2 trials to existing triplets (or one new group)
    if remaining:
        if not triplets:
            # No triplets yet; just one small group with 1–2 elements
            triplets.append(list(remaining))
        else:
            for i, tr in enumerate(remaining):
                triplets[i % len(triplets)].append(tr)

        if verbose:
            print(f"[{condition_name}] Attached {len(remaining)} leftover trial(s) to existing triplets.")

    if verbose:
        sizes = [len(t) for t in triplets]
        print(f"[{condition_name}] Final triplet counts: {len(triplets)} groups, sizes={sizes}")

    return triplets

def loto_folds_from_triplets(triplets: List[List[TrialTuple]]):
    folds = []
    for i, test_group in enumerate(triplets):
        train_group = [tr for j, g in enumerate(triplets) if j != i for tr in g]
        folds.append({
            "train": train_group,
            "test":  test_group,
        })
    return folds

def normalise_w_cov(X, w, eps=1e-12):
    """
    Normalize weights so that w^T Sxx w = 1, where
    Sxx is the covariance of X (T x C).

    Returns a rescaled copy of w.
    """
    X = np.asarray(X, float)
    w = np.asarray(w, float).ravel()

    n = X.shape[0]


    Sxx = (X.T @ X) / max(n - 1, 1)   # covariance (ddof=1)
    quad = float(w.T @ Sxx @ w)
    if quad <= eps:
        # degenerate case: don't rescale
        raise ValueError("Cannot normalise weights: quadratic form too small.")  

    scale = 1.0 / np.sqrt(quad)
    return w * scale


def ridge_wx(X: np.ndarray, y: np.ndarray, lam: float, eps: float = 1e-12, normalise: bool = False ) -> np.ndarray:
    """
    Return ridge-regression weights w for y ≈ Xw.

    Same covariance-style formulation as before, but:
    * no softmax
    * optional L2 normalisation if normalise=True (default False).
    """
    # centre like np.cov does
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    n  = Xc.shape[0]
    Sxx = (Xc.T @ Xc) / max(n - 1, 1)      # ddof=1 covariance
    Sxy = (Xc.T @ yc) / max(n - 1, 1)

    A = Sxx + lam * np.eye(Sxx.shape[0])
    w = np.linalg.solve(A, Sxy)

    if normalise:
        w = normalise_w_cov(X, w)
        w = w[:, None]
    return w
    

def corr_with_weights(X: np.ndarray, y: np.ndarray, w: np.ndarray, return_p: bool = False):
    u = X @ w
    u = np.asarray(u).ravel()
    y = np.asarray(y).ravel()

    # constant signals → Pearson undefined
    if u.std(ddof=0) == 0 or y.std(ddof=0) == 0:
        return (np.nan, np.nan) if return_p else np.nan

    r, p = pearsonr(u, y)  # true Pearson r + p-value

    return (float(r), float(p)) if return_p else float(r)

def mse_with_weights(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """
    Mean squared error between y and Xw.
    Assumes X, y already preprocessed by concat_trials.
    """
    y_hat = X @ w
    return float(np.mean((y_hat - y) ** 2))


# ---------------- per-(lag, λ) training step ---------------------------------
def fit_at_shift_lambda(trials: List[dict], shift_samples: int, lam: float, trim_ms: int, normalise: bool = False) -> Tuple[np.ndarray, float]:
    """
    Fit weights on concatenated TRAIN trials aligned with the given shift.
    Returns (w, train_mse, r_train).
    """
    X_tr, y_tr = concat_trials(trials, shift=shift_samples, trim=trim_ms)

    if len(y_tr) == 0:
        return np.zeros(trials[0]['X'].shape[1]), np.nan
    #w_zoo = ridge_cca_wx_zoo(X_tr, y_tr, lam)

    w_ridge = ridge_wx(X_tr, y_tr, lam, normalise=normalise)
    train_mse = mse_with_weights(X_tr, y_tr, w_ridge)

    r_train = corr_with_weights(X_tr, y_tr, w_ridge)
    return w_ridge, train_mse, r_train

def evaluate_on_trials(trials: List[dict], shift_samples: int, w: np.ndarray, trim_ms: int) -> float:
    """Correlation on a (TRAIN or TEST) split using fixed shift and weights."""
    X_te, y_te = concat_trials(trials, shift=shift_samples, trim=trim_ms)
    if len(y_te) == 0:
        return np.nan
    #return corr_with_weights(X_te, y_te, w)
    return mse_with_weights(X_te, y_te, w), corr_with_weights(X_te, y_te, w)

def lag_corr_curve(trials, lam, shifts, trim_ms):
    """
    Compute correlation vs lag for a fixed λ on a given set of trials.
    """
    rs = []
    for s in shifts:
        _, _, r, _ = fit_at_shift_lambda(trials, int(s), lam, trim_ms)
        rs.append(r)
    return np.array(shifts) * 10, np.array(rs)   # convert to ms


def cross_validate_lambda(lambdas: List[float], folds: List[dict], trials_memory: List[dict], subj: int, trim_ms: int, shifts: np.ndarray, normalise_w: bool = False):
    cv_summaries = []
    
    for lam in lambdas:
        print(f"  Lambda {lam}...")
        fold_rows = []
        train_rss, test_rss = [], []
        train_mses, test_mses = [], []

        for fold_idx, fold in enumerate(folds):
            print(f"  Fold {fold_idx+1}/{len(folds)}...")
            train_trials  = fold["train"]
            test_trials   = fold["test"]
            
            # 1) search best lag on TRAIN for this λ (MAXIMISE train correlation)
            best_shift, best_train_mse, rs_best, best_w = None, np.inf, -np.inf, None
            for s in shifts:
                w_s, mse_tr, rs_tr = fit_at_shift_lambda(train_trials, int(s), lam, trim_ms, normalise=normalise_w)
                
                if np.isfinite(rs_tr) and rs_tr > rs_best:
                    best_train_mse, best_shift, rs_best, best_w = mse_tr, int(s), rs_tr, w_s
                
            # 2) evaluate on TEST using best (lag, w)
            mse_te, rs_te = evaluate_on_trials(test_trials, best_shift, best_w, trim_ms)

            fold_rows.append({
                "lam": lam, "shift": best_shift,
                "train_mse": best_train_mse, "test_mse": mse_te,
                "r_train": rs_best, "r_test": rs_te,
                "n_train": len(train_trials), "n_test": len(test_trials),
            })
            train_rss.append(rs_best);  test_rss.append(rs_te)
            train_mses.append(best_train_mse); test_mses.append(mse_te)

        cv_summaries.append({
            "lam": lam,
            "mean_train_r": float(np.nanmean(train_rss)),
            "mean_test_r": float(np.nanmean(test_rss)),
            "mean_train_mse": float(np.nanmean(train_mses)),
            "mean_test_mse": float(np.nanmean(test_mses)),
            "per_fold": fold_rows,
        })

    print("\n[Subject", subj, "] CV summary by lambda:")
    for cs in cv_summaries:
        print(f"  λ={cs['lam']:>6} | "
            f"train_r={cs['mean_train_r']:+.3f} | "
            f"test_r={cs['mean_test_r']:+.3f} | "
            f"train_mse={cs['mean_train_mse']:.4f} | "
            f"test_mse={cs['mean_test_mse']:.4f}")


    # pick λ with LOWEST mean TEST MSE (tie-breaker: smaller λ)
    ordered_results = sorted(cv_summaries, key=lambda c: (c["mean_test_mse"], c["lam"]))
    best = ordered_results[0]
    print(f"\n[Subject {subj}] Best λ = {best['lam']}, mean_test_mse = {best['mean_test_mse']:.4f}, mean_test_r = {best['mean_test_r']:+.3f}")


    # ------- recompute DEFINITIVE lag + weights at the chosen λ on ALL trials
    rs_shifts, mse_shifts = [], []
    best_shift_all, best_mse_all, best_rs_all, best_w_all = None, np.inf, -np.inf, None
    for s in shifts:
        w_s, mse_all, rs_all = fit_at_shift_lambda(trials_memory, int(s), best["lam"], trim_ms, normalise=normalise_w)
        rs_shifts.append(rs_all)
        mse_shifts.append(mse_all)

        if np.isfinite(rs_all) and rs_all > best_rs_all:
            best_mse_all, best_shift_all, best_rs_all, best_w_all = mse_all, int(s), rs_all, w_s

    

    final_fit = {
        "lambda": best["lam"],
        "best_shift": best_shift_all,
        "w": best_w_all,
        "train_mse_all": best_mse_all,
        "train_rs_all": best_rs_all,
        "cv_mean_test_mse": best["mean_test_mse"],
        "cv_mean_test_r": best["mean_test_r"],
        "cv_mean_train_mse": best["mean_train_mse"],
        "cv_mean_train_r": best["mean_train_r"],
        "cv_details": best["per_fold"],
        "shifts": [int(s) for s in shifts],
        "rs_shifts": rs_shifts,
        "mse_shifts": mse_shifts,
    }
    return final_fit, ordered_results

def fit_best_shift_on_train(train_trials, lam, shifts, trim_ms, normalise_w=False):
    """
    For fixed lambda: choose lag ONLY using training data (parameter estimation).
    Criterion: maximise training correlation (same as speech).
    """
    best_shift, best_w = None, None
    best_r = -np.inf
    best_mse = np.inf

    for s in shifts:
        w_s, mse_tr, r_tr = fit_at_shift_lambda(train_trials, int(s), float(lam), trim_ms, normalise=normalise_w)
        if np.isfinite(r_tr) and r_tr > best_r:
            best_r = r_tr
            best_mse = mse_tr
            best_shift = int(s)
            best_w = w_s

    return best_shift, best_w, best_mse, best_r

def eelbrain_partitions_indices(n_cases: int, k: int):
    """Round-robin partitions: part i contains indices i, i+k, i+2k, ..."""
    k = int(min(k, n_cases))
    return [list(range(i, n_cases, k)) for i in range(k)]


def make_rr_partitions_folds(trials, k: int):
    """Round-robin partitions like eelbrain_partitions_indices, returned as folds=[{train,test}, ...]."""
    n = len(trials)
    parts = eelbrain_partitions_indices(n, k)
    folds = []
    for j in range(len(parts)):
        test_idx = parts[j]
        train_idx = [i for pj in range(len(parts)) if pj != j for i in parts[pj]]
        folds.append({
            "train": [trials[i] for i in train_idx],
            "test":  [trials[i] for i in test_idx],
            "fold":  j,
            "test_idx": test_idx,
        })
    return folds

def choose_load_like_eelbrain(trials_memory):
    """Replicate my_eelbrain_c.py load-selection logic."""
    load_05, load_09, load_13 = split_trials_by_load(trials_memory)
    n5, n9, n13 = len(load_05), len(load_09), len(load_13)

    chosen_load = 13
    chosen_trials = list(load_13)

    if n13 == 0:
        candidates = [(5, n5, list(load_05)), (9, n9, list(load_09))]
        candidates = [(l, n, tr) for (l, n, tr) in candidates if n > 0]
        if not candidates:
            return None, []
        chosen_load, _, chosen_trials = max(candidates, key=lambda t: t[1])
    else:
        if n9 > 0 and n9 >= 1.33 * n13:
            chosen_load, chosen_trials = 9, list(load_09)
        elif n5 > 0 and n5 >= 2.0 * n13:
            chosen_load, chosen_trials = 5, list(load_05)

    return chosen_load, chosen_trials

def nested_cv_lambda_digit(
    trials, lambdas, shifts, trim_ms,
    outer_partitions=4, inner_partitions=3,
    normalise_w=False
):
    # Outer folds (K_out = 4)
    outer_folds = make_rr_partitions_folds(trials, k=outer_partitions)

    outer_rows = []
    y_test_all = []
    yhat_test_all = []

    for o_idx, ofold in enumerate(outer_folds):
        outer_train = ofold["train"]
        outer_test  = ofold["test"]

        # ---------------- Inner CV on OUTER TRAIN: pick lambda ----------------
        # Inner folds are built ONLY from outer_train (K_in = 3 => 2 train + 1 val)
        inner_folds = make_rr_partitions_folds(outer_train, k=inner_partitions)

        inner_summ = []
        for lam in lambdas:
            val_mses = []
            val_rs   = []

            for ifold in inner_folds:
                inner_train = ifold["train"]
                inner_val   = ifold["test"]

                # (parameter estimation) choose lag+weights on inner_train only
                best_shift, best_w, train_mse, train_r = fit_best_shift_on_train(
                    inner_train, lam, shifts, trim_ms, normalise_w=normalise_w
                )

                # score on validation
                X_val, y_val = concat_trials(inner_val, shift=best_shift, trim=trim_ms)
                if len(y_val) == 0:
                    mse_val, r_val = np.nan, np.nan
                else:
                    yhat_val = X_val @ best_w
                    mse_val  = float(np.mean((yhat_val - y_val) ** 2))
                    r_val    = float(pearsonr(np.asarray(yhat_val).ravel(), np.asarray(y_val).ravel())[0])

                val_mses.append(mse_val)
                val_rs.append(r_val)

            inner_summ.append({
                "lam": float(lam),
                "mean_val_mse": float(np.nanmean(val_mses)),
                "mean_val_r":   float(np.nanmean(val_rs)),
            })

        # choose lambda by lowest mean validation MSE (tie-break: smaller lambda)
        lam_star = sorted(inner_summ, key=lambda d: (d["mean_val_mse"], d["lam"]))[0]["lam"]

        # ---------------- Refit on FULL outer_train with lam_star ----------------
        best_shift, best_w, train_mse, train_r = fit_best_shift_on_train(
            outer_train, lam_star, shifts, trim_ms, normalise_w=normalise_w
        )

        # ---------------- Predict on outer_test; store for recombined metric ----------------
        X_te, y_te = concat_trials(outer_test, shift=best_shift, trim=trim_ms)
        if len(y_te) == 0:
            test_mse, test_r = np.nan, np.nan
        else:
            yhat_te = X_te @ best_w
            test_mse = float(np.mean((yhat_te - y_te) ** 2))
            test_r   = float(pearsonr(np.asarray(yhat_te).ravel(), np.asarray(y_te).ravel())[0])

            y_test_all.append(y_te)
            yhat_test_all.append(yhat_te)

        outer_rows.append({
            "outer_fold": int(o_idx),
            "lam_star": float(lam_star),
            "best_shift": int(best_shift),
            "train_mse": float(train_mse),
            "train_r": float(train_r),
            "test_mse_fold": float(test_mse),
            "test_r_fold": float(test_r),
        })

    # ---------------- Recombined OUTER-test metric (dec.r analogue) ----------------
    if len(y_test_all) == 0:
        test_mse_recombined, test_r_recombined = np.nan, np.nan
    else:
        y_all    = np.vstack(y_test_all)
        yhat_all = np.vstack(yhat_test_all)

        test_mse_recombined = float(np.mean((yhat_all - y_all) ** 2))
        test_r_recombined   = float(pearsonr(np.asarray(yhat_all).ravel(), np.asarray(y_all).ravel())[0])

    fold_mean_r   = float(np.nanmean([r["test_r_fold"] for r in outer_rows]))
    fold_mean_mse = float(np.nanmean([r["test_mse_fold"] for r in outer_rows]))

    return {
        "outer_details": outer_rows,
        "test_r_recombined": test_r_recombined,     # <-- compare this to dec.r
        "test_mse_recombined": test_mse_recombined,
        "test_r_fold_mean": fold_mean_r,
        "test_mse_fold_mean": fold_mean_mse,
    }

