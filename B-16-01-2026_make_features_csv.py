import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import welch
from joblib import Parallel, delayed

NPZ_DIR = "/content/processed"
OUT_CSV = "/content/baseline_csv/sleep_night_features_8.csv"

FS = 256
EPOCH_SECONDS = 30
EXPECTED_SAMPLES = FS * EPOCH_SECONDS
EXPECTED_CHANNELS = 8

NON_SPECTRAL_CHANNELS = [0, 1] 

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "sigma": (12, 15),
    "beta":  (15, 30),
}

CHANNELS = [f"ch{i}" for i in range(EXPECTED_CHANNELS)]
EPS = 1e-12

def bandpower_relative(sig_1d, fs=FS):
    freqs, psd = welch(sig_1d, fs=fs, nperseg=fs * 4)

    total_mask = (freqs >= 0.5) & (freqs <= 30.0)
    total_power = np.trapz(psd[total_mask], freqs[total_mask]) + EPS

    out = {}
    for band_name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        bp = np.trapz(psd[mask], freqs[mask])
        out[f"rel_{band_name}"] = float(bp / total_power)

    return out

def time_features(sig_1d):
    std = float(np.std(sig_1d))
    rms = float(np.sqrt(np.mean(sig_1d ** 2)))
    line_len = float(np.sum(np.abs(np.diff(sig_1d))))
    return {"std": std, "rms": rms, "line_len": line_len}

def epoch_channel_features(sig_1d, channel_idx):
    feats = {}
    feats.update(time_features(sig_1d))

    if channel_idx not in NON_SPECTRAL_CHANNELS:
        feats.update(bandpower_relative(sig_1d))
        
    return feats

def night_features_from_epochs(epochs_3d):
    epoch_rows = []

    for ep in epochs_3d:
        row = {}
        for c in range(EXPECTED_CHANNELS):
            feats = epoch_channel_features(ep[c], channel_idx=c)
            
            for k, v in feats.items():
                row[f"{CHANNELS[c]}_{k}"] = v
        epoch_rows.append(row)

    df_epoch = pd.DataFrame(epoch_rows)

    night = {}
    for col in df_epoch.columns:
        night[f"{col}_mean"] = float(df_epoch[col].mean())
        night[f"{col}_std"] = float(df_epoch[col].std(ddof=0))
    return night

def process_single_npz(p):

    try:
        npz = np.load(p, allow_pickle=True)
        if "epochs" not in npz.files:
            return None

        epochs = npz["epochs"]

        if epochs.ndim != 3: return None
        if epochs.shape[1] != EXPECTED_CHANNELS: return None

        label = int(npz["label"]) if "label" in npz.files else None
        label_str = str(npz["label_str"]) if "label_str" in npz.files else None
        filename = str(npz["filename"]) if "filename" in npz.files else None

        feats = night_features_from_epochs(epochs)

        feats["npz_file"] = os.path.basename(p)
        feats["edf_filename"] = filename
        feats["label"] = label
        feats["label_str"] = label_str
        
        return feats

    except Exception as e:
        print(f"Error processing {os.path.basename(p)}: {e}")
        return None

def main():
    paths = sorted(glob.glob(os.path.join(NPZ_DIR, "*.npz")))
    if not paths:
        raise RuntimeError(f"No .npz files found in: {NPZ_DIR}")
    
    print(f"Found {len(paths)} .npz files. starting parallel feature extraction.")
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_single_npz)(p) for p in paths
    )

    rows = [r for r in results if r is not None]

    if not rows:
        print("No valid data extracted.")
        return

    df = pd.DataFrame(rows)

    label_cols = [c for c in ["label", "label_str"] if c in df.columns]
    meta_cols = [c for c in ["npz_file", "edf_filename"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in meta_cols + label_cols]
    df = df[meta_cols + feature_cols + label_cols]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved CSV: {OUT_CSV}")
    print("Shape:", df.shape)

    if "label_str" in df.columns:
        print("\nClass counts:")
        print(df["label_str"].value_counts())

if __name__ == "__main__":
    main()