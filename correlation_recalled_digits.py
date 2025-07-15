# %%
import pandas as pd
from pathlib import Path
from scipy import stats

# where you saved the files
CORR_FILE   = Path(r"C:\Users\cdd\Documents\Uni\Special_course\df.csv")  
BEH_DIR    = Path(r"C:\Users\cdd\Documents\Uni\Special_course\ds003838-download")

SUBJECTS   = ["sub-034"]      # put all subjects here when you’re ready


# %%
corr = (pd.read_csv(CORR_FILE)
          .assign(                       # extract trial #, then +1 to match *.tsv
              trial=lambda d:
                  d.epoch.str.extract(r"trial_(\d+)\.csv")
                    .astype(int).squeeze()
                    + 1                 # 0-based → 1-based
          ))


# %%
def explode_triggers(df_beh):
    out_frames = []
    for _, row in df_beh.iterrows():
        trig = row["triggerCorrect"].strip()
        out_frames.append(pd.DataFrame({
            "trial"     : row["trial"],
            "digit_pos" : range(1, len(trig) + 1),   # 1, 2, …
            "recalled"  : [int(c) for c in trig]      # 0 / 1
        }))
    return pd.concat(out_frames, ignore_index=True)


# %%
subject_dfs = []      # collect for an optional grand-average
stats_per_sub = []    # store per-subject t-tests

for sub in corr["subject"].unique():
    # --- Behaviour ------------------------------------------------------
    beh_path = (BEH_DIR / sub / "beh" / f"{sub}_task-memory_beh.tsv")
    beh_raw  = pd.read_csv(beh_path, sep="\t", usecols=["trial", "triggerCorrect"], dtype = {"triggerCorrect": str})
    beh      = explode_triggers(beh_raw)

    # --- Merge ----------------------------------------------------------
    merged = corr.query("subject == @sub").merge(
                 beh, on=["trial", "digit_pos"], how="inner")

    # --- Stats ----------------------------------------------------------
    rec  = merged.loc[merged.recalled == 1, "r"]
    miss = merged.loc[merged.recalled == 0, "r"]

    t, p = stats.ttest_ind(rec, miss, equal_var=False)
    stats_per_sub.append(
        {"subject": sub,
         "mean_r_recalled": rec.mean(),
         "mean_r_missed"  : miss.mean(),
         "t" : t, "p" : p, "n_recalled": len(rec), "n_missed": len(miss)}
    )

    merged["subject"] = sub
    subject_dfs.append(merged)


# %%
pd.set_option("display.precision", 3)
print(pd.DataFrame(stats_per_sub))

# %%
print(f"sub {sub}:  n_recalled = {len(rec)},  var = {rec.var(ddof=1):.3g}")
print(f"sub {sub}:  n_missed   = {len(miss)}, var = {miss.var(ddof=1):.3g}")


# %%
import pandas as pd
from scipy import stats

# concatenate everything the loop stored
group_df = pd.concat(subject_dfs, ignore_index=True)

# two samples
rec_all  = group_df.loc[group_df.recalled == 1, "r"]
miss_all = group_df.loc[group_df.recalled == 0, "r"]

# Welch two-sample t-test → one-tailed (recalled > missed)
t_stat, p_two = stats.ttest_ind(rec_all, miss_all, equal_var=False, nan_policy="omit")
p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2

print(f"Pooled: t = {t_stat:.3f}, one-tailed p = {p_one:.4g}, "
      f"mean Δr = {rec_all.mean() - miss_all.mean():.3f}  (N = {len(rec_all)+len(miss_all)})")


# %%
# 1) build a table of means per subject × recalled(0/1)
sub_means = (group_df
             .groupby(["subject", "recalled"])["r"]
             .mean()
             .unstack())        # columns 0 = missed, 1 = recalled

# 2) drop any subject lacking one category (all-correct or all-wrong)
sub_means = sub_means.dropna(subset=[0, 1])

# 3) paired t-test (recalled − missed)
from scipy.stats import ttest_rel
t_stat, p_two = ttest_rel(sub_means[1], sub_means[0])
p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2

print(f"Paired: t = {t_stat:.3f}, one-tailed p = {p_one:.4g}, "
      f"mean Δr = {(sub_means[1]-sub_means[0]).mean():.3f}  (N_subjects = {len(sub_means)})")



