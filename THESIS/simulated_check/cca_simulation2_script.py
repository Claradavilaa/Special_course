# %% [markdown]
# More simple.  just gt EEG component, and weightd

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt


# ===================== config =====================
FS = 100  # Hz
FRONTAL_MIDLINE = ['AFz','AF3','AF4','Fz','F1','F2','F3','F4','FC3','FC1','FC2','FC4','Cz','C3','C1','C2','C4']
N_CH = len(FRONTAL_MIDLINE)
RNG = np.random.default_rng(7)

# ===================== helpers ====================
import numpy as np

def _power(x):
    return float(np.mean(x**2))

def snr_db_component_projection(tr):
    """
    SNR of the 1-D component you'd like CCA to recover.
    Signal:   c(t) * ||w||^2   (so X@w matches scale in noiseless case)
    Noise:    (X@w) - (c*||w||^2)
    """
    X, c, w = tr['X'], tr['c'], tr['w_true']
    comp_obs  = X @ w
    comp_true = c * float(w @ w)
    Ps, Pn = _power(comp_true), _power(comp_obs - comp_true)
    return 10.0 * np.log10(max(Ps, 1e-20) / max(Pn, 1e-20))

def snr_db_pupil(tr, fs, lag_ms, trim_ms=300):
    """
    SNR of the pupil versus the delayed ground-truth component.
    Signal: delayed c(t)
    Noise:  y - delayed c(t)
    """
    y = tr['y']
    # build delayed component (same convention as your generator)
    c_del = shift_signal(tr['c'], fs, lag_ms)
    T = min(len(y), len(c_del))
    y = y[:T]; c_del = c_del[:T]
    # optional edge trim to avoid pad artifacts
    k = int(round((trim_ms or 0) / (1000.0/fs)))
    if k > 0 and T > 2*k:
        y = y[k:-k]; c_del = c_del[k:-k]
    Ps, Pn = _power(c_del), _power(y - c_del)
    return 10.0 * np.log10(max(Ps, 1e-20) / max(Pn, 1e-20))

def snr_db_channels(tr, crosstalk_lambda=0.0, treat_crosstalk_as_signal=True):
    """
    Per-channel SNRs.
    Signal:    X_true = outer(c, w_true)
               (+ optional leakage term if treat_crosstalk_as_signal)
    Noise:     X - Signal
    Returns:   array shape (C,)
    """
    X, c, w = tr['X'], tr['c'], tr['w_true']
    X_true = np.outer(c, w)                      # ideal (noiseless) channels
    if treat_crosstalk_as_signal and crosstalk_lambda > 0:
        mean_t = X_true.mean(axis=1, keepdims=True)
        leak   = crosstalk_lambda * (mean_t - X_true)
        S = X_true + leak
    else:
        S = X_true
    N = X - S
    Ps = np.mean(S**2, axis=0)
    Pn = np.mean(N**2, axis=0)
    return 10.0 * np.log10(np.maximum(Ps, 1e-20) / np.maximum(Pn, 1e-20))

### ============ pupil component split =================
import numpy as np
from scipy.signal import butter, filtfilt

# --- your filter, extended to 1D/2D (time along axis 0)
def _butter_filt(x, fs, kind="low", fc=(0.25,)):
    if kind == "low":
        b, a = butter(4, fc[0] / (fs/2), btype="low")
    elif kind == "band":
        b, a = butter(4, [fc[0] / (fs/2), fc[1] / (fs/2)], btype="band")
    else:
        raise ValueError("kind must be 'low' or 'band'")

    x = np.asarray(x)
    padlen = None
    # choose a safe padlen per filtfilt docs
    base_pad = 3 * max(len(a), len(b))
    if x.ndim == 1:
        padlen = min(base_pad, len(x) - 1)
        return filtfilt(b, a, x, padlen=padlen)

    # x.ndim == 2: filter each column independently (T, C)
    T = x.shape[0]
    padlen = min(base_pad, T - 1)
    out = np.empty_like(x, dtype=float)
    for c in range(x.shape[1]):
        out[:, c] = filtfilt(b, a, x[:, c], padlen=padlen)
    return out

# --- split components for one trial dict from simulate_component_dataset
def split_components_one_trial(trial, fs, low_fc=0.25, band=(0.4, 0.7)):
    """
    Adds to `trial`:
      y_slow, y_fast, X_slow, X_fast
    where 'slow' = low-pass(low_fc), 'fast' = band-pass(band).
    Returns a shallow-copied dict with new keys.
    """
    y = np.asarray(trial['y'], dtype=float)         # (T,)
    X = np.asarray(trial['X'], dtype=float)         # (T, C)

    y_slow = _butter_filt(y, fs, kind="low",  fc=(low_fc,))
    y_fast = _butter_filt(y, fs, kind="band", fc=band)

    X_slow = _butter_filt(X, fs, kind="low",  fc=(0.2,))
    X_fast = _butter_filt(X, fs, kind="band", fc=band)

    out = dict(trial)  # shallow copy
    out.update(dict(
        y_slow=y_slow, y_fast=y_fast,
        X_slow=X_slow, X_fast=X_fast
    ))
    return out

# --- convenience wrapper for a whole list of trials
def split_components_all_trials(trials, fs, low_fc=0.25, band=(0.4, 0.7)):
    return [split_components_one_trial(tr, fs, low_fc=low_fc, band=band) for tr in trials]

import numpy as np
import matplotlib.pyplot as plt

def _time_axis(trial, fs):
    T = len(trial['y'])
    return np.arange(T) / float(fs)

# ---- EEG (X) slow/fast for selected channels
def plot_X_components(trial, fs, ch_idx=(0,1,2,3), ch_names=None, title_prefix="Trial"):
    """
    ch_idx: tuple/list of channel indices to plot.
    ch_names: optional list of names for all channels (len = C).
    """
    X      = np.asarray(trial['X'])
    X_slow = np.asarray(trial['X_slow'])
    X_fast = np.asarray(trial['X_fast'])
    t = _time_axis(trial, fs)

    ch_idx = list(ch_idx)
    n = len(ch_idx)
    plt.figure(figsize=(11, 2.5*n))

    for i, c in enumerate(ch_idx, start=1):
        name = ch_names[c] if (ch_names is not None and c < len(ch_names)) else f"ch{c}"
        ax1 = plt.subplot(n, 2, 2*i-1)
        ax1.plot(t, X[:, c], label=f'{name} raw')
        ax1.plot(t, X_slow[:, c], label=f'{name} slow (LP)')
        ax1.set_ylabel('Amp')
        if i == 1: ax1.set_title(f"{title_prefix} — EEG raw vs. slow")
        ax1.legend(loc='best')

        ax2 = plt.subplot(n, 2, 2*i)
        ax2.plot(t, X_fast[:, c], label=f'{name} fast (band)')
        if i == n: ax2.set_xlabel('Time (s)')
        if i == 1: ax2.set_title(f"{title_prefix} — EEG fast (band)")
        ax2.legend(loc='best')

    plt.tight_layout()
    plt.show()

def plot_y_vs_proj_components(
    trial, fs, w=None, title_prefix="Trial"
):
    """
    Plots:
      1) y_slow vs proj_slow
      2) y_fast vs proj_fast

    Expects trial dict with keys: 'y', 'y_slow', 'y_fast', 'X', optionally 'X_slow', 'X_fast', 'c', 'w_true'.
    """
    X = np.asarray(trial['X'], dtype=float)
    y = np.asarray(trial['y'], dtype=float)
    y_slow = np.asarray(trial.get('y_slow', y), dtype=float)
    y_fast = np.asarray(trial.get('y_fast', y*0), dtype=float)

    if w is None:
        w = np.asarray(trial.get('w_true'))
        if w is None:
            raise ValueError("Provide w or include 'w_true' in the trial dict.")
    w = w.astype(float)

    T = len(y)
    t = np.arange(T) / float(fs)

    # use precomputed filtered matrices if available
    if ('X_slow' not in trial) or ('X_fast' not in trial):
        raise ValueError("X_slow/X_fast not found in trial; set use_project_then_filter=True or compute them first.")
    proj_slow = np.asarray(trial['X_slow']) @ w
    proj_fast = np.asarray(trial['X_fast']) @ w

    # --- Plot
    fig = plt.figure(figsize=(10, 6))
    ax1 = plt.subplot(2,1,1)
    ax1.plot(t, y_slow, label='y_slow (LP)')
    ax1.plot(t, proj_slow, label='proj_slow (X_slow@w)')
    if 'c' in trial:
        ax1.plot(t, trial['c'], label='ground-truth c', alpha=0.6)
    ax1.plot(t, y, label='y (raw)', alpha=0.6)
    ax1.set_title(f"{title_prefix} — Slow components")
    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='best')

    ax2 = plt.subplot(2,1,2)
    ax2.plot(t, y_fast, label='y_fast (band)')
    ax2.plot(t, proj_fast, label='proj_fast (X_fast@w)')
    ax2.set_title(f"{title_prefix} — Fast components")
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.show()



def zscore1(x):
    m, s = x.mean(), x.std(ddof=0)
    return (x - m) / (s if s > 0 else 1.0)

def normalise_eeg(X: np.ndarray) -> np.ndarray:
    """Per-trial: demean each channel, then scale by single global RMS (timexchannels)."""
    X = X - X.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(X**2))
    return X / (scale if scale > 0 else 1.0)

def shift_signal(x, fs, lag_ms):
    """If lag_ms < 0, signal is delayed"""
    s = int(np.round(lag_ms * fs / 1000.0))
    if s == 0: return x.copy()
    if s < 0: 
        s = -s 
        return np.r_[np.full(s, x[0]), x[:-s]]
    else:      
        return np.r_[x[s:], np.full(s, x[-1])]

def align_by_shift_segment(X, y, shift_samples):
    """Drop-edge alignment (no wrap). Positive shift => drop early EEG, drop late pupil."""
    T = min(len(X), len(y))
    if shift_samples >= 0:
        s = shift_samples
        if T <= s: return None, None
        return X[s:], y[:T - s]
    else:
        s = -shift_samples
        if T <= s: return None, None
        return X[:T - s], y[s:]

def concat_trials(trials, shift_samples=0, trim_ms=None):
    """Align (drop-edge), optional trim, per-trial normalize (RMS for X, zscore for y), then stack."""
    Xs, Ys = [], []
    k = int(round((trim_ms or 0) / 10))  # 10 ms/sample at 100 Hz
    for tr in trials:
        X, y = tr['X'], tr['y']
        Xs_al, ys_al = align_by_shift_segment(X, y, shift_samples)
        if Xs_al is None: 
            continue
        if k > 0:
            if len(ys_al) <= 2*k: 
                continue
            Xs_al = Xs_al[k:-k]
            ys_al = ys_al[k:-k]
        Xs_al = normalise_eeg(Xs_al)  # per-trial, like you asked
        ys_al = zscore1(ys_al)
        Xs.append(Xs_al); Ys.append(ys_al)
    if not Xs:
        return np.empty((0, trials[0]['X'].shape[1])), np.empty((0,))
    return np.vstack(Xs), np.concatenate(Ys)

def cca_corr(X, y):
    cca = CCA(n_components=1, max_iter=1000, scale=False, tol=1e-6)
    cca.fit(X, y.reshape(-1,1))
    u, v = cca.transform(X, y.reshape(-1,1))
    return float(pearsonr(u[:,0], v[:,0])[0])

def search_best_lag(train_trials, shifts, trim_ms=None, return_curve=False):
    r_per_shift = {}
    for s in shifts:
        X, y = concat_trials(train_trials, shift_samples=s, trim_ms=trim_ms)
        if len(y) == 0:
            r_per_shift[s] = np.nan
            continue
        r_per_shift[s] = cca_corr(X, y)
    # pick best over valid shifts
    valid = {k: v for k, v in r_per_shift.items() if np.isfinite(v)}
    best_s = max(valid, key=valid.get)
    best_r = valid[best_s]
    if return_curve:
        return best_r, best_s, r_per_shift
    return best_r, best_s, None

def search_best_lag_trialwise(trials, shifts, trim_ms=None, agg="mean", return_curve=False):
    """Compute r per-trial at each shift (after per-trial norm), then aggregate across trials."""
    r_per_shift = {}
    for s in shifts:
        rs = []
        for tr in trials:
            X, y = tr['X'], tr['y']
            Xs, ys = align_by_shift_segment(X, y, s)
            if Xs is None: 
                continue
            k = int(round((trim_ms or 0)/10))
            if k > 0:
                if len(ys) <= 2*k: 
                    continue
                Xs, ys = Xs[k:-k], ys[k:-k]
            Xs = normalise_eeg(Xs)
            ys = zscore1(ys)
            if len(ys) >= 5:
                rs.append(cca_corr(Xs, ys))
        r_per_shift[s] = (np.nan if len(rs)==0 
                          else (np.nanmean(rs) if agg=="mean" else np.nanmedian(rs)))
    valid = {k:v for k,v in r_per_shift.items() if np.isfinite(v)}
    best_s = max(valid, key=valid.get)
    best_r = valid[best_s]
    return (best_r, best_s, r_per_shift) if return_curve else (best_r, best_s, None)

def candidate_lags_units(step_ms=10, max_ms=1000):
    steps_ms = np.arange(-max_ms, max_ms + step_ms, step_ms)
    return (steps_ms / 10).astype(int)

# ============== component-level simulator ==============
def make_gt_component(T_sec, fs, kind="tasky"):
    """Ground-truth EEG component c(t) to be recovered."""
    T = int(round(T_sec*fs))
    t = np.arange(T)/fs
    if kind == "tasky":
        # smooth on/plateau/off + a little wiggle
        slow_on = 0.5*np.tanh((t-2.0)/1.2) - 0.5*np.tanh((t-(T_sec-2.0))/1.2)
        plateau = 0.6 + 0.4*np.tanh((t-6.0)/2.0) - 0.4*np.tanh((t-(T_sec-6.0))/2.0)
        amp = 0.6*slow_on + 0.4*plateau
        wiggle = 0.15*np.sin(2*np.pi*0.5*t)  # sub-1 Hz wiggle in "power"
        c = amp + wiggle
    elif kind == "ou":
        # Ornstein–Uhlenbeck-like smooth noise
        x = RNG.normal(size=T)
        alpha = np.exp(-1.0/(fs*0.5))  # ~0.5 s time-const
        c = np.zeros(T); 
        for i in range(1, T): c[i] = alpha*c[i-1] + (1-alpha)*x[i]
        c = (c - c.min()); c /= (c.max() if c.max()>0 else 1)  # positive
    else:
        c = np.ones(T)
    return zscore1(c)  # standardize for convenience

def simulate_component_dataset_old(
    n_trials=12, trial_len_s=26.0,
    pupil_lag_ms=350,
    w_spread=0.5,
    eeg_noise_sd=0.15,
    eeg_crosstalk=0.10,
    pupil_noise_sd=0.10,
    w_true_fixed= None,   #  keep same weights across runs
):
    """
    Returns trials with:
      X: (T,C) per-channel theta 'power' (linear mixes of component),
      y: (T,) pupil = shifted component + noise,
      c: (T,) ground-truth component,
      w_true: (C,) ground-truth weights (unit norm).
    If w_true_fixed is provided, ALL trials share those weights.
    """
    trials = []

    if w_true_fixed is None:
        w = 1.0 + w_spread*RNG.normal(size=N_CH)
        w_true = w / (np.linalg.norm(w) or 1.0)
    else:
        w_true = w_true_fixed / (np.linalg.norm(w_true_fixed) or 1.0)

    for _ in range(n_trials):
        c = make_gt_component(trial_len_s, FS, kind="tasky")
        T = len(c)
        X = np.outer(c, w_true)

        if eeg_crosstalk > 0:
            leak = eeg_crosstalk * (X.mean(axis=1, keepdims=True) - X)
            X = X + leak

        if eeg_noise_sd != 0:
            C = X.shape[1]
            sigma_c = eeg_noise_sd * (1 + 0.3*RNG.normal(size=C))
            X += RNG.normal(size=X.shape) * sigma_c

        y = shift_signal(c, FS, pupil_lag_ms)
        if pupil_noise_sd != 0:
            y = y + pupil_noise_sd * RNG.normal(size=T)

        trials.append({'X': X, 'y': y, 'c': c, 'w_true': w_true})
    return trials

def simulate_component_dataset(
    n_triplets: int = 4,
    triplet_lengths_s: tuple[float, float, float] = (26.0, 18.0, 5.0),
    pupil_lag_ms: int = 350,
    w_spread: float = 0.5,
    eeg_noise_sd: float = 0.15,
    eeg_crosstalk: float = 0.10,
    pupil_noise_sd: float = 0.10,
    w_true_fixed: np.ndarray = None,
    shuffle_within_triplet: bool = False,
    rng: np.random.Generator = None,   # NEW
):
    """
    Returns trials grouped in triplets. Each triplet contains exactly one trial
    of each length in 'triplet_lengths_s'. Adds metadata: 'triplet_id', 'length_s'.
    """
    trials = []
    rng = rng or RNG

    # choose / normalize GT weights once
    if w_true_fixed is None:
        w = 1.0 + w_spread * rng.normal(size=N_CH)
        w_true = w / (np.linalg.norm(w) or 1.0)
    else:
        w_true = w_true_fixed / (np.linalg.norm(w_true_fixed) or 1.0)

    for triplet_id in range(n_triplets):
        lengths = list(triplet_lengths_s)
        if shuffle_within_triplet:
            rng.shuffle(lengths)
        for L in lengths:
            c = make_gt_component(L, FS, kind="tasky")
            T = len(c)
            X = np.outer(c, w_true)

            if eeg_crosstalk > 0:
                leak = eeg_crosstalk * (X.mean(axis=1, keepdims=True) - X)
                X = X + leak

            if eeg_noise_sd != 0:
                C = X.shape[1]
                sigma_c = eeg_noise_sd * (1 + 0.3 * rng.normal(size=C))
                X = X + rng.normal(size=X.shape) * sigma_c

            y = shift_signal(c, FS, pupil_lag_ms)
            if pupil_noise_sd != 0:
                y = y + pupil_noise_sd * rng.normal(size=T)

            trials.append({
                'X': X, 'y': y, 'c': c, 'w_true': w_true,
                'triplet_id': triplet_id, 'length_s': float(L)
            })

    # sanity check: each triplet has exactly the requested lengths
    from collections import Counter, defaultdict
    by_trip = defaultdict(list)
    for i, tr in enumerate(trials):
        by_trip[tr['triplet_id']].append(tr['length_s'])
    for tid, lens in by_trip.items():
        assert Counter(lens) == Counter(triplet_lengths_s), \
            f"Triplet {tid} does not contain exactly one of each length."

    return trials


## WITH NOISE:
def ar1_noise(T, phi=0.98, sd=1.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    e = rng.normal(scale=sd, size=T)
    x = np.empty(T); x[0] = e[0]
    for t in range(1, T): x[t] = phi*x[t-1] + e[t]
    return (x - x.mean()) / (x.std(ddof=0) or 1)

def pink_noise(T, rng=None):
    # crude 1/f via filtered white
    rng = np.random.default_rng() if rng is None else rng
    w = rng.normal(size=T)
    # long moving average subtract → slow-heavy residual
    slow = np.convolve(w, np.ones(401)/401, mode='same')
    x = w - slow
    return (slow / (slow.std(ddof=0) or 1))

def random_walk(T, step_sd=0.01, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    steps = rng.normal(scale=step_sd, size=T)
    x = np.cumsum(steps)
    return (x - x.mean()) / (x.std(ddof=0) or 1)

def independent_slow_driver(T, kind='ar1', rng=None):
    if kind=='ar1':   return ar1_noise(T, phi=0.995, sd=1.0, rng=rng)
    if kind=='pink':  return pink_noise(T, rng=rng)
    if kind=='walk':  return random_walk(T, step_sd=0.005, rng=rng)
    raise ValueError

def sat_nonlinearity(x, k=1.2):
    return np.tanh(k*x)       # compressive, slow remains

def time_varying_weights(w_true, T, drift_sd=0.15, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    drift = ar1_noise(T, phi=0.995, sd=drift_sd, rng=rng)[:,None]   # slow scalar
    delta = rng.normal(size=w_true.shape)                           # random dir
    w_t = w_true + drift * (delta/np.linalg.norm(delta))
    return w_t  # shape (T, C)

def jittered_shift(c, fs, base_lag_ms=-350, jitter_ms=120, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    lag = base_lag_ms + rng.integers(-jitter_ms, jitter_ms+1)
    return shift_signal(c, fs, lag), lag

def simulate_component_dataset_noise(
    n_trials=12, trial_len_s=26.0, pupil_lag_ms=0,
    w_spread=0.5, eeg_noise_sd=0.15, eeg_crosstalk=0.10, pupil_noise_sd=0.10,
    w_true_fixed=None,
    # NEW knobs:
    pupil_extra=None,          # None|'ar1'|'pink'|'walk'
    pupil_extra_scale=0.5,      # amplitude of extra slow driver
    pupil_nonlinear=False,      # apply tanh on pupil signal
    lag_jitter_ms=0,            # e.g. 100 for ±100 ms per-trial jitter
    eeg_weight_drift=False,     # allow slow drift in EEG mixing
    eeg_weight_drift_sd=0.15,
):
    trials = []
    rng = RNG  # use your seeded RNG

    if w_true_fixed is None:
        w = 1.0 + w_spread*rng.normal(size=N_CH)
        w_true = w / (np.linalg.norm(w) or 1.0)
    else:
        w_true = w_true_fixed / (np.linalg.norm(w_true_fixed) or 1.0)

    for _ in range(n_trials):
        c = make_gt_component(trial_len_s, FS, kind="tasky")
        T = len(c)

        # ---- EEG generation with optional slow weight drift
        if eeg_weight_drift:
            w_t = time_varying_weights(w_true, T, drift_sd=eeg_weight_drift_sd, rng=rng)  # (T,C)
            X = (c[:,None] * w_t)
        else:
            X = np.outer(c, w_true)

        if eeg_crosstalk > 0:
            leak = eeg_crosstalk * (X.mean(axis=1, keepdims=True) - X)
            X = X + leak

        if eeg_noise_sd != 0:
            C = X.shape[1]
            sigma_c = eeg_noise_sd * (1 + 0.3*rng.normal(size=C))
            X += rng.normal(size=X.shape) * sigma_c

        # ---- pupil path with optional lag jitter
        if lag_jitter_ms:
            y_base, _ = jittered_shift(c, FS, base_lag_ms=pupil_lag_ms, jitter_ms=lag_jitter_ms, rng=rng)
        else:
            y_base = shift_signal(c, FS, pupil_lag_ms)

        # add extra slow driver (independent)
        if pupil_extra in ('ar1','pink','walk'):
            d = independent_slow_driver(T, kind=pupil_extra, rng=rng)
            y_base = y_base + pupil_extra_scale * d

        if pupil_nonlinear:
            y_base = sat_nonlinearity(y_base, k=1.2)

        if pupil_noise_sd != 0:
            y_base = y_base + pupil_noise_sd * rng.normal(size=T)

        trials.append({'X': X, 'y': y_base, 'c': c, 'w_true': w_true})
    return trials




"""

# %%
# ======================= demo ========================
if __name__ == "__main__":
    TRUE_LAG_MS = -350
    trials = simulate_component_dataset(
        n_trials=12, trial_len_s=26.0,
        pupil_lag_ms=TRUE_LAG_MS,
        w_spread=0.7, eeg_noise_sd=0.3, eeg_crosstalk=0.3, pupil_noise_sd=0.4,
    )

    # After you simulate `trials` (before any concat/normalization):

    # 1) Component-projection SNR (one per trial)
    snr_comp = [snr_db_component_projection(tr) for tr in trials]

    # 2) Pupil SNR (use the same lag you used to generate y)
    snr_pup = [snr_db_pupil(tr, FS, TRUE_LAG_MS, trim_ms=300) for tr in trials]

    # 3) Channel-level SNRs (vector per trial). If you used eeg_crosstalk=0.10:
    LAM = 0.10
    snr_ch = [snr_db_channels(tr, crosstalk_lambda=LAM, treat_crosstalk_as_signal=False) for tr in trials]

    print(f"Component SNR (dB):  mean={np.mean(snr_comp):.2f}, median={np.median(snr_comp):.2f}")
    print(f"Pupil     SNR (dB):  mean={np.mean(snr_pup):.2f},  median={np.median(snr_pup):.2f}")
    print(f"Channel   SNR (dB):  median-of-medians={np.median([np.median(a) for a in snr_ch]):.2f}")


    # --- lag search (trial-wise aggregation so every trial has 1 vote) ---
    shifts = candidate_lags_units(step_ms=10, max_ms=1000)
    best_r, best_shift_samp, curve = search_best_lag(trials, shifts, trim_ms=300, return_curve=True)

    best_shift_ms = int(best_shift_samp*10)
    print(f"GT lag = {TRUE_LAG_MS} ms | recovered = {best_shift_ms} ms | r = {best_r:.3f}")

    # --- fit CCA once at best lag on all per-trial-normalized segments ---
    X_all, y_all = concat_trials(trials, shift_samples=best_shift_samp, trim_ms=300)
    cca = CCA(n_components=1, max_iter=1000, scale=False, tol=1e-6)
    cca.fit(X_all, y_all.reshape(-1,1))
    w_est = cca.x_weights_[:,0]
    w_est = w_est / (np.linalg.norm(w_est) if np.linalg.norm(w_est)>0 else 1.0)

    # --- compare weights (use trial 1's w_true as representative; they are same across sims) ---
    w_true = trials[0]['w_true']
    cos_sim = float(np.dot(w_true, w_est) / (np.linalg.norm(w_true)*np.linalg.norm(w_est)))
    print(f"Weight alignment (cosine) = {cos_sim:.3f}  (sign/scale are arbitrary)")

    # --- plot r vs lag ---
    xs = np.array(sorted(curve.keys()))*10
    ys = np.array([curve[k] for k in sorted(curve.keys())])
    plt.figure(figsize=(6.2,3.2))
    plt.plot(xs, ys, lw=2)
    plt.axvline(TRUE_LAG_MS, ls=':', label=f"true = {TRUE_LAG_MS} ms")
    plt.axvline(best_shift_ms, ls='--', label=f"best = {best_shift_ms} ms")
    plt.xlabel("Lag (ms) [EEG → PPD]")
    plt.ylabel("Canonical r (trial-avg)")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()

    # --- pick one trial ---
    tr = trials[0]
    X1, y1, c1 = tr['X'], tr['y'], tr['c']
    s = best_shift_samp
    K = int(round(300/10))  # same trim as training

    # 1) Build the ALIGNED set (for recovered comp & pupil)
    X1_al, y1_al = align_by_shift_segment(X1, y1, s)
    X1_al, y1_al = X1_al[K:-K], y1_al[K:-K]
    X1_al = normalise_eeg(X1_al)
    y1_al = zscore1(y1_al)

    # Recovered component (aligned)
    w_est = cca.x_weights_[:, 0]
    c_rec = zscore1(X1_al @ w_est)

    # 2) Build the UNALIGNED set with the SAME LENGTH (no shift)
    #    Take the corresponding segment of X1 and c1 that matches the aligned length,
    #    but WITHOUT shifting them in time.
    Tmn = min(len(X1), len(y1))
    if s >= 0:
        # aligned kept X[:Tmn-s], so keep the same slice unshifted
        X1_un = X1[:Tmn - s]
        c1_un = c1[:Tmn - s]
    else:
        # aligned kept X[|s|:], so keep the same slice unshifted
        X1_un = X1[-s:]
        c1_un = c1[-s:]

    # apply the same trim K (but no alignment shift)
    X1_un = X1_un[K:-K]
    c1_un = c1_un[K:-K]

    # per-trial normalization for plotting consistency
    X1_un = normalise_eeg(X1_un)

    # True projection (UNALIGNED)
    w_true = tr['w_true']
    c_true_proj_un = zscore1(X1_un @ w_true)

    # 3) Plot
    t = np.arange(len(y1_al)) / FS
    plt.figure(figsize=(9.5,4.0))
    plt.plot(t, y1_al,            label="Pupil (aligned, z)")
    plt.plot(t, zscore1(c1_un),             label="GT component c(t)", alpha=.9)
    plt.plot(t, c_rec,            label="Recovered X·w_CCA (aligned, z)")
    plt.plot(t, c_true_proj_un,   label="True X·w_true (UNALIGNED, z)")
    # Optionally also show c_al if you want GT aligned view:
    # plt.plot(t, zscore1(c_al), label="GT c(t) (aligned, z)", ls="--")
    plt.xlabel("Time (s)"); plt.ylabel("z-scored amplitude")
    plt.title("Aligned recovered component vs UNALIGNED true projection")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()


    # --- weights bar plot ---
    plt.figure(figsize=(9.6,4.1))
    idx = np.arange(N_CH)
    plt.bar(idx-0.2, w_true/np.linalg.norm(w_true), width=0.4, label="true (normed)")
    plt.bar(idx+0.2, w_est/np.linalg.norm(w_est),   width=0.4, label="CCA (normed)")
    plt.xticks(idx, FRONTAL_MIDLINE, rotation=45)
    plt.ylabel("Weight (normalized)"); 
    plt.title(f"Electrode weights — true vs recovered (cosine = {cos_sim:.3f})")
    plt.legend(); plt.tight_layout()
    plt.show()


# %%
# ---------------- experiment: n_trials sweep ----------------
TRUE_LAG_MS = -350  # your convention: negative = delay

# Fix one GT weight vector across ALL runs so cos-sim is comparable
w0 = 1.0 + 0.7 * RNG.normal(size=N_CH)
w0 = w0 / (np.linalg.norm(w0) or 1.0)

trial_counts = [1] + list(range(10, 110, 10)) + list(range(125, 225, 25))

cos_sims = []
best_rs  = []
best_lags = []
anomalies = []  # (n_trials, best_r, best_shift_ms)

for n in trial_counts:
    trials = simulate_component_dataset(
        n_trials=n, trial_len_s=26.0,
        pupil_lag_ms=TRUE_LAG_MS,
        w_spread=0.7, eeg_noise_sd=0.3, eeg_crosstalk=0.3, pupil_noise_sd=0.4,
        w_true_fixed=w0,  # keep same GT weights across runs
    )

    # 1) lag search on concatenated, per-trial-normalized data
    shifts = candidate_lags_units(step_ms=10, max_ms=1000)
    best_r, best_shift_samp, _ = search_best_lag(
        trials, shifts, trim_ms=300, return_curve=True
    )
    best_shift_ms = int(best_shift_samp * 10)

    # 2) fit CCA once at best lag and get weights
    X_all, y_all = concat_trials(trials, shift_samples=best_shift_samp, trim_ms=300)
    cca = CCA(n_components=1, max_iter=1000, scale=False, tol=1e-6)
    cca.fit(X_all, y_all.reshape(-1, 1))
    w_est = cca.x_weights_[:, 0]

    # cosine similarity (sign-agnostic)
    num = float(np.dot(w0, w_est))
    den = (np.linalg.norm(w0) or 1.0) * (np.linalg.norm(w_est) or 1.0)
    cs = abs(num / den)

    cos_sims.append(cs)
    best_rs.append(best_r)
    best_lags.append(best_shift_ms)

    if best_shift_ms != TRUE_LAG_MS:
        anomalies.append((n, best_r, best_shift_ms))

# ---- report anomalies ----
if anomalies:
    print("⚠️ Cases where recovered lag ≠ GT lag:")
    for n, r, lag in anomalies:
        print(f"  n_trials={n:3d}: best_r={r:.3f}, best_shift_ms={lag} (GT {TRUE_LAG_MS})")
else:
    print("✅ All recovered lags matched the ground-truth.")

# ---- plot: cos_sim vs n_trials ----
plt.figure(figsize=(6.4, 3.4))
plt.plot(trial_counts, cos_sims, marker='o', lw=1.8)
plt.ylim(0.0, 1.02)
plt.xlabel("Number of trials")
plt.ylabel("Cosine(w_true, w_est)")
plt.title("Weight recovery vs number of trials")
plt.grid(alpha=.3)
plt.tight_layout()

# ---- plot: best_r vs n_trials ----
plt.figure(figsize=(6.4, 3.4))
plt.plot(trial_counts, best_rs, marker='o', lw=1.8)
plt.ylim(0.0, 1.02)
plt.xlabel("Number of trials")
plt.ylabel("Best canonical r")
plt.title("Best CCA correlation vs number of trials")
plt.grid(alpha=.3)
plt.tight_layout()
plt.show()



"""