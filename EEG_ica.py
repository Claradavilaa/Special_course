"""
batch_amica_theta.py  –  full AMICA → Fig 5c workflow for all subjects
---------------------------------------------------------------------

* Saves every diagnostic (correlation table, ICA plots, 1-channel noise log)
* Produces cleaned FIFs and the theta-band replication figure
"""

import os, sys, logging, pathlib
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from scipy.stats import sem

# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------
BASE_DIR   = r"C:\Users\cdd\Documents\Uni\Special_course\ds003838-download"
OUT_DIR    = r"C:\Users\cdd\Documents\Uni\Special_course\code\Special_course\EEG_amica_processed"
OUT_DIR    = pathlib.Path(OUT_DIR)
OUT_DIR.mkdir(exist_ok=True)

SUBJECTS   = np.setdiff1d(np.arange(32, 99), [37, 53, 66, 94, 96])
# SUBJECTS   = np.setdiff1d(np.arange(32, 99), [37, 66, 94, 96])
RESAMPLE   = 250           # Hz  (leave None to keep original)
THRESHOLD  = 0.30          # |corr| threshold for EOG detection
ROI        = ['Fz']        # electrode(s) for the theta curve
# ---------------------------------------------------------------------
FREQS      = np.arange(1, 46)
N_CYCLES   = np.logspace(np.log10(3), np.log10(12), len(FREQS))
BASELINE   = (-2., -1.)
BIN_DUR    = 2.0
LISTLEN_MAP = {5: 11., 9: 17., 13: 27.}
X_POS      = np.arange(1, 14)
# ---------------------------------------------------------------------
logging.basicConfig(
    filename=OUT_DIR / 'batch_log.txt',
    encoding='utf-8',        # <── add this
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s: %(message)s',
)


def theta_power(tfr, beg, end):
    f_sel = np.where((FREQS >= 4) & (FREQS <= 8))[0]
    return tfr.copy().crop(beg, end).data[:, f_sel, :].mean()

def first_digit_ids(ids, cond_prefix, list_len):
    ll = f"{list_len:02d}"
    return {c:e for c,e in ids.items()
            if c.startswith(cond_prefix) and c[2:4]=='01' and c[4:6]==ll}

# ---------------------------------------------------------------------
# RUN SUBJECT BY SUBJECT
# ---------------------------------------------------------------------
mem_all, ctrl_all = [], []

for subj in SUBJECTS:
    sub_tag = f"sub-{subj:03d}"
    raw_set = (pathlib.Path(BASE_DIR) / sub_tag / "eeg" / f"{sub_tag}_task-memory_eeg.set")
    sub_out = OUT_DIR / sub_tag
    sub_out.mkdir(exist_ok=True)
    clean_fif = sub_out / f"{sub_tag}_task-memory_eeg_amica.fif"

    if not raw_set.exists():
        logging.warning("%s – missing EEGLAB file, skipped", sub_tag)
        continue

    print(f"▶  {sub_tag}")
    if clean_fif.exists():
        print(f"   {sub_tag} – already processed, skipping ICA")
        logging.info("%s – already ICA, skipped ICA part", sub_tag)
        raw_clean = mne.io.read_raw_fif(clean_fif, preload=True, verbose='error')

    else:
        logging.info("%s – processing ICA", sub_tag)
        # ------------------------------------------------ ICA / CLEAN STEP
        raw = mne.io.read_raw_eeglab(raw_set, preload=True, verbose='error')
        raw.filter(1., 45., fir_design='firwin', verbose='error')
        raw.set_eeg_reference('average', verbose='error')
        if RESAMPLE: raw.resample(RESAMPLE, verbose='error')

        # bipolar pseudo-EOG
        raw = mne.set_bipolar_reference(raw, 'Fp1','AFz', ch_name='VEOG',
                                        drop_refs=False, copy=False)
        raw = mne.set_bipolar_reference(raw, 'F7','F8',  ch_name='HEOG',
                                        drop_refs=False, copy=False)
        raw.set_channel_types({'VEOG':'eog', 'HEOG':'eog'})

        ica = ICA(method='infomax', n_components=0.99, random_state=97,
                fit_params=dict(extended=True, max_iter=800))
        try:
            ica.fit(raw, reject_by_annotation=True)

        except RuntimeError as err: # subject 63
            if "One PCA component captures most" in str(err):
                print(f"{sub_tag}: extreme variance imbalance - refitting ICA with n_components=20")
                logging.warning("%s - PCA 1-component problem: refit with 20 PCs", sub_tag)

                ica = ICA(method='infomax', n_components=20, random_state=97,
                        fit_params=dict(extended=True, max_iter=800))
                ica.fit(raw, reject_by_annotation=True)
            else:
                raise        # re-throw unknown errors

        ica.fit(raw, reject_by_annotation=True)

        eog_inds, eog_scores = ica.find_bads_eog(
            raw, ['VEOG','HEOG'], measure='correlation', threshold=THRESHOLD)
        ica.exclude = eog_inds.copy()

        # ---------- save styled correlation table -----------------------
        corr_df = pd.DataFrame({
            'VEOG': eog_scores[0],
            'HEOG': eog_scores[1]
        })
        corr_df.index = [f'IC {i:02d}' for i in range(len(eog_scores[0]))]
        corr_df = corr_df.T

        styled = corr_df.style.map(
            lambda v: 'background-color: yellow; font-weight:bold'
                    if abs(v) >= THRESHOLD else '')
        styled.to_html(sub_out / f"{sub_tag}_eog_corr.html")

        # ---------- 1-channel sensor noise check ------------------------
        lines = []
        topo = ica.get_components()     # (n_chan, n_comp)
        ch_names = np.array(raw.ch_names)

        for k in range(topo.shape[1]):
            w = np.abs(topo[:, k])
            if (w > .2 * w.max()).sum() == 1:
                max_idx = w.argmax()
                energy = (w[max_idx]**2 / (w**2).sum()) * 100
                line = (f"IC {k:02d}: {ch_names[max_idx]:>4s} carries "
                        f"{energy:5.1f}% of weight (|w|={w[max_idx]:.2e}) -> 1-chan noise")
                print("   ", line)
                lines.append(line)
                ica.exclude.append(k)
        # write log
        with open(sub_out / f"{sub_tag}_one-chan-noise.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) if lines else "NONE")


        # ---------- save ICA figures ------------------------------------
        if not ica.exclude == []:
            fig = ica.plot_components(picks=ica.exclude, show=False)
            fig.savefig(sub_out / f"{sub_tag}_ica_components.png", dpi=150)
            plt.close(fig)

            # ----- save diagnostic figures -------------
            if eog_inds:                     # <-- guard against empty list
                prop_figs = ica.plot_properties(raw, picks=eog_inds, show=False)
                for pf, ic in zip(prop_figs, eog_inds):
                    pf.savefig(sub_out / f"{sub_tag}_ica_prop_IC{ic:02d}.png", dpi=150)
                    plt.close(pf)
            else:
                logging.info("%s – no EOG ICs found at threshold %.2f", sub_tag, THRESHOLD)

        # ---------- apply & save cleaned data ---------------------------
        raw_clean = ica.apply(raw.copy())
        raw_clean.save(clean_fif, overwrite=True)

    # ------------------------------------------------ TFR / FIG 5c STEP
    logging.info("%s – processing TFR", sub_tag)
    events, ids = mne.events_from_annotations(raw_clean)
    mem_by_pos  = [[] for _ in range(13)]
    ctrl_by_pos = [[] for _ in range(13)]

    for n_digits, tmax in LISTLEN_MAP.items():
        for prefix, bucket in [('60', mem_by_pos), ('50', ctrl_by_pos)]:
            first_ids = first_digit_ids(ids, prefix, n_digits)
            if not first_ids: continue

            ep = mne.Epochs(raw_clean, events, event_id = first_ids,
                            tmin=-2., tmax=tmax, baseline=None,
                            picks='eeg', preload=True, verbose='error')

            ep = mne.preprocessing.compute_current_source_density(ep, copy=True)

            tfr = mne.time_frequency.tfr_morlet(
                ep.pick_channels(ROI), freqs=FREQS, n_cycles=N_CYCLES,
                return_itc=False, verbose='error').apply_baseline(BASELINE,'percent')

            for pos in range(1, n_digits+1):
                beg, end = BIN_DUR*(pos-1), BIN_DUR*pos
                bucket[pos-1].append(theta_power(tfr, beg, end))

    mem_all.append([np.nanmean(v) if v else np.nan for v in mem_by_pos])
    ctrl_all.append([np.nanmean(v) if v else np.nan for v in ctrl_by_pos])

    # per-subject curve ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(X_POS, mem_all[-1],  '-o', label='Memory')
    ax.plot(X_POS, ctrl_all[-1], '-s', label='Control')
    ax.set(title=f'{sub_tag}', xlabel='Serial position', ylabel='Theta (% vs BL)')
    ax.legend(frameon=False); ax.set_ylim(bottom=0)
    plt.tight_layout()
    (OUT_DIR / 'sub-plots').mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / 'sub-plots' / f"{sub_tag}_theta.png", dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------
# GRAND-AVERAGE PLOT
# ---------------------------------------------------------------------
mem_all  = np.asarray(mem_all,  float)
ctrl_all = np.asarray(ctrl_all, float)

plt.figure(figsize=(9,5))
plt.errorbar(X_POS, np.nanmean(mem_all,axis=0),
             yerr=sem(mem_all,axis=0,nan_policy='omit'),
             marker='o',label='Memory',linewidth=2)
plt.errorbar(X_POS, np.nanmean(ctrl_all,axis=0),
             yerr=sem(ctrl_all,axis=0,nan_policy='omit'),
             marker='s',label='Control',linewidth=2)
plt.xlabel('Serial position (digit)')
plt.ylabel('Theta power (% change vs baseline)')
plt.title('Replication of Kosachenko et al. 2023 — θ band (all subjects)')
plt.xticks(X_POS); plt.ylim(bottom=0); plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(OUT_DIR / 'theta_replication.png', dpi=300)
plt.show()

print("\n✅  Done.  Everything is in", OUT_DIR)
