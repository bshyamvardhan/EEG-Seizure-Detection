import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cdist
import os
# ==============================
# CONFIGURATION
# ==============================
fs = 173.61
base_path = 'Base_Dataset'

folders = ['S','F','N','Z','O']
labels  = ['ictal','interictal','interictal','healthy','healthy']

feature_matrix = []
class_labels = []

# ==============================
# FILTER DESIGN
# ==============================
def bandpass(low, high, fs, order=4):
    return butter(order, [low/(fs/2), high/(fs/2)], btype='band')

b_bp, a_bp = bandpass(0.5, 40, fs)
b_delta, a_delta = bandpass(0.5, 4, fs)
b_theta, a_theta = bandpass(4, 8, fs)
b_alpha, a_alpha = bandpass(8, 13, fs)
b_beta,  a_beta  = bandpass(13, 30, fs)
b_gamma, a_gamma = bandpass(30, 40, fs)

# ==============================
# SAMPLE ENTROPY
# ==============================
def sampen(x, m, r):
    x = np.array(x)
    N = len(x)

    if N <= m + 1:
        return np.nan

    def _embed(x, dim):
        return np.array([x[i:N-dim+i+1] for i in range(dim)]).T

    Xm = _embed(x, m)
    Xm1 = _embed(x, m+1)

    D = cdist(Xm, Xm, metric='chebyshev')
    D1 = cdist(Xm1, Xm1, metric='chebyshev')

    B = np.sum(D < r, axis=1) - 1
    A = np.sum(D1 < r, axis=1) - 1

    B_total = np.sum(B)
    A_total = np.sum(A)

    if A_total == 0 or B_total == 0:
        return np.inf
    else:
        return -np.log(A_total / B_total)

# ==============================
# APPROXIMATE ENTROPY
# ==============================
def approximate_entropy(x, m, r):
    x = np.array(x)
    N = len(x)

    def _phi(m):
        X = np.array([x[i:N-m+i+1] for i in range(m)]).T
        C = np.sum(cdist(X, X, 'chebyshev') <= r, axis=0) / (N - m + 1)
        return np.sum(np.log(C)) / (N - m + 1)

    return _phi(m) - _phi(m+1)

# ==============================
# BANDPOWER
# ==============================
def bandpower(signal, fs, band):
    f, Pxx = welch(signal, fs=fs)
    idx = np.logical_and(f >= band[0], f <= band[1])
    from scipy.integrate import trapezoid
    return trapezoid(Pxx[idx], f[idx])

# ==============================
# MAIN LOOP
# ==============================
for f_idx, folder in enumerate(folders):

    folder_path = os.path.join(base_path, folder)
    if not os.path.isdir(folder_path):
        print(f"Missing folder: {folder_path}")
        continue

    files = [file for file in os.listdir(folder_path) if file.lower().endswith('.txt')]
    if not files:
        print(f"No .txt files in {folder_path}")
        continue

    for file in files:

        eeg = np.loadtxt(os.path.join(folder_path, file))
        eeg = eeg.flatten()

        # PREPROCESS
        eeg = eeg - np.mean(eeg)
        eeg_filt = filtfilt(b_bp, a_bp, eeg)

        # SUBBANDS
        delta = filtfilt(b_delta, a_delta, eeg_filt)
        theta = filtfilt(b_theta, a_theta, eeg_filt)
        alpha = filtfilt(b_alpha, a_alpha, eeg_filt)
        beta  = filtfilt(b_beta,  a_beta,  eeg_filt)
        gamma = filtfilt(b_gamma, a_gamma, eeg_filt)

        # ==========================
        # FEATURES
        # ==========================

        # (A) Statistical
        stat_feats = [
            np.mean(eeg_filt),
            np.std(eeg_filt),
            np.var(eeg_filt),
            skew(eeg_filt),
            kurtosis(eeg_filt)
        ]

        # (B) Band Power
        band_feats = [
            bandpower(delta, fs, [0.5, 4]),
            bandpower(theta, fs, [4, 8]),
            bandpower(alpha, fs, [8, 13]),
            bandpower(beta,  fs, [13, 30]),
            bandpower(gamma, fs, [30, 40])
        ]

        # (C) Entropy
        shannon_ent = -np.sum((np.abs(eeg_filt)**2) * np.log2(np.abs(eeg_filt)**2 + 1e-12))
        log_energy  = np.sum(np.log(eeg_filt**2 + 1e-12))

        approx_ent = approximate_entropy(eeg_filt, 2, 0.2*np.std(eeg_filt))
        sample_ent = sampen(eeg_filt, 2, 0.2*np.std(eeg_filt))

        f_psd, psd = welch(eeg_filt, fs)
        psd = psd / np.sum(psd)
        spectral_ent = -np.sum(psd * np.log2(psd + 1e-12))

        entropy_feats = [
            shannon_ent,
            log_energy,
            approx_ent,
            sample_ent,
            spectral_ent
        ]

        feature_vector = stat_feats + band_feats + entropy_feats

        if len(feature_vector) != 15:
            raise ValueError("Feature length mismatch")

        feature_matrix.append(feature_vector)
        class_labels.append(labels[f_idx])

# ==============================
# SAVE DATASET
# ==============================
columns = [
    'Mean','Std','Variance','Skewness','Kurtosis',
    'DeltaPower','ThetaPower','AlphaPower','BetaPower','GammaPower',
    'ShannonEntropy','LogEnergyEntropy','ApproxEntropy','SampleEntropy','SpectralEntropy'
]

df = pd.DataFrame(feature_matrix, columns=columns)
df['Class'] = class_labels

df.to_csv('Feature_Extraction/EEG_Feature_Dataset.csv', index=False)

print("✅ PHASE-2 COMPLETED SUCCESSFULLY")
print(f"📊 Total samples: {len(df)}")