import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Path to: ...\THESIS\ridge_reg
ridge_reg_dir = Path(r"C:\Users\cdd\Documents\Uni\Special_course\code\Special_course\THESIS\ridge_reg")

# Add it so Python can import modules from there
if str(ridge_reg_dir) not in sys.path:
    sys.path.insert(0, str(ridge_reg_dir))


from ridge_regression_functions import *

###############################################################################
# Paths & constants
###############################################################################
#PUPIL_ROOT = Path(r"C:\Users\cdd\Documents\Uni\Special_course\pupil_fast")
#EEG_ROOT   = Path(r"C:\Users\cdd\Documents\Uni\Special_course\code\Special_course\eeg_fast")

PUPIL_ROOT = Path(r"C:\Users\cdd\Documents\Uni\Special_course\pupil_processed_clara")
EEG_ROOT    = Path(r"C:\Users\cdd\Documents\Uni\Special_course\code\Special_course\eeg_theta_processed2")

FRONTAL_MIDLINE = ['AFz','AF3','AF4','Fz','F1','F2','F3','F4','FC3','FC1','FC2','FC4','Cz','C3','C1','C2','C4']
SHIFTS = candidate_lags_units(step_ms=100, max_ms=1500)

SUBJECTS = np.setdiff1d(np.arange(32, 99), [32, 37, 53, 61, 66, 78, 84, 90, 94, 96])

# -----------------------------
# Helpers (eelbrain-style CV + refit-all r)
# -----------------------------
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

def eelbrain_partitions_indices(n_cases: int, k: int):
    """Round-robin partitions: part i contains indices i, i+k, i+2k, ..."""
    k = int(min(k, n_cases))
    return [list(range(i, n_cases, k)) for i in range(k)]


def predict_on_trials(trials, shift_samples, w, trim_ms):
    X, y = concat_trials(trials, shift=int(shift_samples), trim=trim_ms)
    if len(y) == 0:
        return np.array([]), np.array([])
    y_hat = (X @ w).ravel()
    return y_hat, np.asarray(y).ravel()


def evaluate_on_trials(trials, shift_samples, w, trim_ms):
    """Returns (mse, r) on these trials."""
    yhat, y = predict_on_trials(trials, shift_samples, w, trim_ms)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() == 0:
        return np.nan, np.nan
    yv, yhatv = y[mask], yhat[mask]
    mse = float(np.mean((yv - yhatv) ** 2))
    r = float(np.corrcoef(yv, yhatv)[0, 1]) if mask.sum() > 1 else np.nan
    return mse, r


def r_manual_refit_all(trials, shift, lam, trim_ms):
    """Refit on ALL trials, then manual corr(y, X@w) on ALL samples (in-sample)."""
    w, _, _ = fit_at_shift_lambda(trials, int(shift), float(lam), trim_ms)
    X_all, y_all = concat_trials(trials, shift=int(shift), trim=trim_ms)

    yhat = (X_all @ w).ravel()
    y = np.asarray(y_all).ravel()

    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() <= 1:
        return np.nan, np.nan, w

    yv, yhatv = y[mask], yhat[mask]
    r = float(np.corrcoef(yv, yhatv)[0, 1])

    resid = yv - yhatv
    var_y = np.var(yv, ddof=0)
    var_res = np.var(resid, ddof=0)
    prop_expl = np.nan if var_y == 0 else 1.0 - (var_res / var_y)
    prop_expl_pct = 100.0 * prop_expl if np.isfinite(prop_expl) else np.nan

    return r, prop_expl_pct, w




# -----------------------------
# CV where lambda is hyperparameter, shift is treated like a fitted parameter
# (chosen on TRAIN only inside each outer fold)
# -----------------------------
def cv_lam_only_shift_as_param(
    trials,
    lambdas,
    shifts,
    trim_ms,
    partitions=4,
    choose_shift_by="train_r",         # "train_r" or "train_mse"
    select_lambda_by="mean_test_mse",  # "mean_test_mse" or "mean_test_r"
):
    n = len(trials)
    if n < 2:
        return np.nan, pd.DataFrame(), {}, []

    parts = eelbrain_partitions_indices(n, partitions)
    k = len(parts)

    per_lambda = {float(lam): {"test_r": [], "test_mse": [], "chosen_shift": []} for lam in lambdas}
    fold_details = []  # one entry per outer fold per lambda (nice for saving)

    for test_pi in range(k):
        test_trials  = [trials[i] for i in parts[test_pi]]
        train_trials = [trials[i] for pi in range(k) if pi != test_pi for i in parts[pi]]

        for lam in lambdas:
            lam = float(lam)

            # ---- choose shift on TRAIN only ----
            best_s = None
            best_score = -np.inf if choose_shift_by == "train_r" else np.inf
            best_w = None
            best_train_r = np.nan
            best_train_mse = np.nan

            for s in shifts:
                w, _, _ = fit_at_shift_lambda(train_trials, int(s), lam, trim_ms)
                mse_tr, r_tr = evaluate_on_trials(train_trials, int(s), w, trim_ms)

                score = r_tr if choose_shift_by == "train_r" else mse_tr
                if not np.isfinite(score):
                    continue

                better = (score > best_score) if choose_shift_by == "train_r" else (score < best_score)
                if better:
                    best_score = score
                    best_s = int(s)
                    best_w = w
                    best_train_r = float(r_tr)
                    best_train_mse = float(mse_tr)

            # ---- evaluate on TEST ----
            mse_te, r_te = evaluate_on_trials(test_trials, int(best_s), best_w, trim_ms)

            per_lambda[lam]["test_r"].append(float(r_te))
            per_lambda[lam]["test_mse"].append(float(mse_te))
            per_lambda[lam]["chosen_shift"].append(int(best_s))

            fold_details.append({
                "test_partition": int(test_pi),
                "lam": float(lam),
                "chosen_shift": int(best_s),
                "train_r_at_chosen_shift": float(best_train_r),
                "train_mse_at_chosen_shift": float(best_train_mse),
                "test_r": float(r_te),
                "test_mse": float(mse_te),
                "n_train_trials": int(len(train_trials)),
                "n_test_trials": int(len(test_trials)),
            })

    # summarize per lambda
    summary_rows = []
    for lam in sorted(per_lambda.keys()):
        mean_r = float(np.nanmean(per_lambda[lam]["test_r"]))
        mean_mse = float(np.nanmean(per_lambda[lam]["test_mse"]))
        summary_rows.append({"lam": float(lam), "mean_test_r": mean_r, "mean_test_mse": mean_mse})

    summary = pd.DataFrame(summary_rows)

    # choose lambda
    if summary.empty:
        lam_star = np.nan
    else:
        if select_lambda_by == "mean_test_r":
            lam_star = float(summary.loc[summary["mean_test_r"].idxmax(), "lam"])
        else:
            lam_star = float(summary.loc[summary["mean_test_mse"].idxmin(), "lam"])

    return lam_star, summary, per_lambda, fold_details


def choose_shift_on_all_data(trials, lam, shifts, trim_ms, choose_shift_by="train_r"):
    """Pick shift using ALL trials (same criterion as inside folds), then fit w on ALL trials."""
    lam = float(lam)

    best_s = None
    best_score = -np.inf if choose_shift_by == "train_r" else np.inf
    best_w = None

    for s in shifts:
        w, _, _ = fit_at_shift_lambda(trials, int(s), lam, trim_ms)
        mse_all, r_all = evaluate_on_trials(trials, int(s), w, trim_ms)

        score = r_all if choose_shift_by == "train_r" else mse_all
        if not np.isfinite(score):
            continue

        better = (score > best_score) if choose_shift_by == "train_r" else (score < best_score)
        if better:
            best_score = score
            best_s = int(s)
            best_w = w

    return best_s, best_w


# -----------------------------
# SUBJECT LOOP using cv_lam_only_shift_as_param
# -----------------------------
SHIFTS = candidate_lags_units(step_ms=50, max_ms=1500)
trim_ms = 200
lambdas = [0, 0.1, 1, np.power(10, 0.5), 10, np.power(10, 2.5)]
NOT_EVAL_SUBS = [35, 46, 60, 68, 97]
PARTITIONS = 4

# choices that match your original intent:
CHOOSE_SHIFT_BY = "train_r"          # shift treated as parameter; pick it on TRAIN by correlation
SELECT_LAMBDA_BY = "mean_test_mse"   # choose lambda by held-out performance (change to mean_test_r if you want)

DETAILS_DIR = Path("cv_details_lam_only_ALL")
DETAILS_DIR.mkdir(exist_ok=True, parents=True)

rows_subject = []

for subj in SUBJECTS:
    if subj in NOT_EVAL_SUBS:
        continue

    print(f"\n========== Subject {subj:02d} ==========")

    # 1) load + filter to memory
    all_trials = load_all_trials(subj, eeg_root=EEG_ROOT, pupil_root=PUPIL_ROOT)
    trials_memory, _ = split_trials_by_condition(all_trials)
    print(f"Loaded {len(trials_memory)} memory trials")

    loads_mem = [meta["load"] for (_, _, meta) in trials_memory]
    print(f"Memory loads counts: 5={loads_mem.count(5)}, 9={loads_mem.count(9)}, 13={loads_mem.count(13)}")

    # 2) choose load like eelbrain
    chosen_load, trials_use = choose_load_like_eelbrain(trials_memory)
    if len(trials_use) < 2:
        print("Not enough trials after load selection - skipping.")
        continue
    print(f"Using load {chosen_load} with {len(trials_use)} trials (eelbrain-style selection)")

    # 3) CV: lambda is hyperparam, shift is fit inside each fold on TRAIN only
    lam_star, lam_summary, per_lambda, fold_details = cv_lam_only_shift_as_param(
        trials_use,
        lambdas=lambdas,
        shifts=SHIFTS,
        trim_ms=trim_ms,
        partitions=PARTITIONS,
        choose_shift_by=CHOOSE_SHIFT_BY,
        select_lambda_by=SELECT_LAMBDA_BY,
    )

    # overall “dec.r analogue” for the chosen lambda (mean over folds or recompute from fold_details)
    if len(fold_details):
        # Mean test r for chosen lambda (matches the selection summaries)
        r_cv_star = float(lam_summary.loc[lam_summary["lam"] == lam_star, "mean_test_r"].iloc[0])
        mse_cv_star = float(lam_summary.loc[lam_summary["lam"] == lam_star, "mean_test_mse"].iloc[0])
    else:
        r_cv_star, mse_cv_star = np.nan, np.nan

    print(f"Chosen λ* = {lam_star}  (by {SELECT_LAMBDA_BY})")
    print(f"CV mean test r at λ* = {r_cv_star:.3f}, mean test mse at λ* = {mse_cv_star:.3f}")

    # 4) After selecting lambda, estimate shift on ALL data (treat shift like parameter) and refit-all
    shift_star, _ = choose_shift_on_all_data(
        trials_use, lam=lam_star, shifts=SHIFTS, trim_ms=trim_ms, choose_shift_by=CHOOSE_SHIFT_BY
    )
    r_refit_all_star, ve_refit_all_star_pct, w_refit_star = r_manual_refit_all(
        trials_use, shift=shift_star, lam=lam_star, trim_ms=trim_ms
    )
    print(f"Refit-all: shift*={shift_star} ({shift_star*10} ms), r_train={r_refit_all_star:.3f}")

    # 5) Also save CV test r for lambda=0 (already included in lam_summary; just extract)
    if (lam_summary["lam"] == 0.0).any():
        r_cv_lam0 = float(lam_summary.loc[lam_summary["lam"] == 0.0, "mean_test_r"].iloc[0])
        mse_cv_lam0 = float(lam_summary.loc[lam_summary["lam"] == 0.0, "mean_test_mse"].iloc[0])
    else:
        r_cv_lam0, mse_cv_lam0 = np.nan, np.nan

    # refit-all for lambda=0 (choose shift on all data, then fit all)
    shift0_star, _ = choose_shift_on_all_data(
        trials_use, lam=0.0, shifts=SHIFTS, trim_ms=trim_ms, choose_shift_by=CHOOSE_SHIFT_BY
    )
    r_refit_all_lam0, ve_refit_all_lam0_pct, w_refit0 = r_manual_refit_all(
        trials_use, shift=shift0_star, lam=0.0, trim_ms=trim_ms
    )

    # 6) Save per-subject details (NOT in dfk)
    subj_prefix = f"sub-{subj:03d}_l{int(chosen_load)}"

    # Summary per lambda
    lam_summary.to_csv(DETAILS_DIR / f"{subj_prefix}_lambda_summary.csv", index=False)

    # Fold-by-fold details (each fold *each lambda*)
    pd.DataFrame(fold_details).to_csv(DETAILS_DIR / f"{subj_prefix}_fold_details.csv", index=False)

    # JSON bundle (handy archive)
    with (DETAILS_DIR / f"{subj_prefix}_bundle.json").open("w", encoding="utf-8") as f:
        json.dump({
            "subject": int(subj),
            "chosen_load": int(chosen_load) if chosen_load is not None else None,
            "n_trials_used": int(len(trials_use)),
            "choose_shift_by": CHOOSE_SHIFT_BY,
            "select_lambda_by": SELECT_LAMBDA_BY,
            "lam_star": float(lam_star),
            "shift_star_samples": int(shift_star),
            "shift_star_ms": int(shift_star) * 10,
            "cv_mean_test_r_at_lam_star": float(r_cv_star),
            "cv_mean_test_mse_at_lam_star": float(mse_cv_star),
            "cv_lambda_summary": lam_summary.to_dict(orient="records"),
        }, f, indent=2)

    # 7) Save dfk row (no giant details)
    rows_subject.append({
        "subject": subj,
        "chosen_load": int(chosen_load) if chosen_load is not None else np.nan,
        "n_trials_used": int(len(trials_use)),

        # CV summaries (hyperparameter selection)
        "lam_star": float(lam_star),
        "cv_mean_test_r_star": float(r_cv_star),
        "cv_mean_test_mse_star": float(mse_cv_star),

        # Final fitted parameter (shift) + refit-all training metrics
        "shift_star_samples": int(shift_star),
        "lag_star_ms": int(shift_star) * 10,
        "r_refit_all_star": float(r_refit_all_star),
        "prop_expl_refit_all_star_%": float(ve_refit_all_star_pct),

        # Baseline lambda=0
        "cv_mean_test_r_lam0": float(r_cv_lam0),
        "cv_mean_test_mse_lam0": float(mse_cv_lam0),
        "shift_lam0_samples": int(shift0_star),
        "lag_lam0_ms": int(shift0_star) * 10,
        "r_refit_all_lam0": float(r_refit_all_lam0),
        "prop_expl_refit_all_lam0_%": float(ve_refit_all_lam0_pct),

        # optional weights
        "w_star": w_refit_star,
        "w_lam0": w_refit0,
    })

    # incremental save
    pd.DataFrame(rows_subject).sort_values("subject").reset_index(drop=True).to_csv(
        "dfk_ridge_eelbrain_style_shift_param_TMP.csv", index=False
    )

# final save
dfk = pd.DataFrame(rows_subject).sort_values("subject").reset_index(drop=True)
dfk.to_csv("dfk_ridge_eelbrain_ALL.csv", index=False)
print("\nSaved dfk_ridge_eelbrain_ALL.csv")
print(f"Saved per-subject details in: {DETAILS_DIR.resolve()}")



