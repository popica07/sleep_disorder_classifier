import os
import glob
import warnings
import gc
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import mne
from joblib import Parallel, delayed


RAW_EDF_DIR = "/content/raw"
PROCESSED_DIR = "/content/processed"

SAMPLE_RATE = 256  
EPOCH_SEC = 30     
EPOCH_SAMPLES = SAMPLE_RATE * EPOCH_SEC

CHANNEL_NAMES = ["PLETH", "SAO2", "F4-C4", "C4-A1", "C4-P4", "P4-O2", "ROC-LOC", "ECG1-ECG2"]

EEG_EOG_ECG = ["F4-C4", "C4-A1", "C4-P4", "P4-O2", "ROC-LOC", "ECG1-ECG2"]

CLASS_NAMES = [
    "bruxism", "insomnia", "narcolepsy", "nocturnal_frontal_lobe_epilepsy",
    "periodic_leg_movements", "rem_behavior_disorder", "sleep_disordered_breathing", "healthy_control"
]
CLASS_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

PREFIX_TO_LABEL = {
    "brux": "bruxism", "ins": "insomnia", "narco": "narcolepsy", "nfle": "nocturnal_frontal_lobe_epilepsy",
    "plm": "periodic_leg_movements", "rbd": "rem_behavior_disorder", "sdb": "sleep_disordered_breathing", "n": "healthy_control"
}

os.makedirs(PROCESSED_DIR, exist_ok=True)
mne.set_log_level("WARNING")

def clean_sao2(x: np.ndarray, fs: float) -> tuple[np.ndarray, dict]:
    x = np.asarray(x, dtype=np.float32).copy()
    n_total = len(x)
    
    drop_thresh = 10.0
    recovery_sec = 5.0
    recovery_tol = 3.0
    lo, hi = 0.0, 100.0

    bad = ~np.isfinite(x) | (x < lo) | (x > hi)

    dx = np.diff(x, prepend=x[0])
    drop_indices = np.where(dx <= -drop_thresh)[0]
    
    if len(drop_indices) > 0:
        recovery_samples = int(max(1, round(recovery_sec * fs)))
        for i in drop_indices:
            if i > 0 and bad[i-1]: continue
            pre_val = x[i-1]
            end_window = min(n_total, i + recovery_samples + 1)
            if not np.any(x[i:end_window] >= (pre_val - recovery_tol)):
                bad[i:end_window] = True

    if np.any(bad):
        valid = np.where(~bad)[0]
        invalid = np.where(bad)[0]
        if valid.size > 0:
            x[invalid] = np.interp(invalid, valid, x[valid])
        else:
            x[:] = 0.0

    x = np.clip(x, lo, hi)
    n_bad = int(bad.sum())
    stats = {"n_imputed": n_bad, "frac_imputed": n_bad / n_total if n_total else 0.0}
    return x, stats

CAP_LABEL_MAP = {
    "SLEEP-S0": 0, "SLEEP-S1": 1, "SLEEP-S2": 2, "SLEEP-S3": 3, "SLEEP-S4": 3, "SLEEP-REM": 4
}
DROP_LABELS = ["SLEEP-UNSCORED", "MT", "Movement", "MCAP-A1", "MCAP-A2", "MCAP-A3"]

def parse_txt_annotations(txt_path, edf_start_time):
    if not os.path.exists(txt_path): return []
    header_idx = -1
    with open(txt_path, 'r', encoding='latin-1') as f:
        for i, line in enumerate(f):
            if line.strip().startswith("Sleep Stage"):
                header_idx = i
                break
    if header_idx == -1: return []

    try:
        df = pd.read_csv(txt_path, skiprows=header_idx, sep='\t', engine='python')
    except: return []

    df.columns = [c.strip() for c in df.columns]
    required_cols = ['Sleep Stage', 'Time [hh:mm:ss]', 'Event', 'Duration[s]']
    if not all(col in df.columns for col in required_cols):
        return []

    valid_epochs = []
    
    for _, row in df.iterrows():
        event = str(row['Event']).strip()
        duration = row['Duration[s]']
        time_str = str(row['Time [hh:mm:ss]']).strip()

        if not event.startswith("SLEEP-"): continue
        if event in DROP_LABELS: continue
        if event not in CAP_LABEL_MAP: continue
        label_int = CAP_LABEL_MAP[event]
            
        try:
            if float(duration) != 30.0: continue
            
            annot_time = datetime.strptime(time_str, '%H:%M:%S')
            onset_dt = edf_start_time.replace(hour=annot_time.hour, minute=annot_time.minute, second=annot_time.second)
            
            if onset_dt < edf_start_time: 
                onset_dt += timedelta(days=1)
            
            onset_sec = (onset_dt - edf_start_time).total_seconds()
            
            if onset_sec >= 0:
                valid_epochs.append((onset_sec, label_int))
        except: continue
            
    return valid_epochs

def process_single_file(edf_path):
    filename = os.path.basename(edf_path)
    out_path = os.path.join(PROCESSED_DIR, os.path.splitext(filename)[0] + ".npz")
    txt_path = edf_path.replace(".edf", ".txt")

    if not os.path.exists(txt_path): return None

    label_str, label_idx = None, None
    name_base = os.path.splitext(filename)[0].lower()
    for prefix in ["brux", "ins", "narco", "nfle", "plm", "rbd", "sdb", "n"]:
        if name_base.startswith(prefix):
            label_str = PREFIX_TO_LABEL[prefix]
            label_idx = CLASS_INDEX[label_str]
            break
    if label_str is None: return None

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")

        raw.rename_channels(lambda s: s.strip().upper())
        
        desired = [ch.upper() for ch in CHANNEL_NAMES]
        if not set(desired).issubset(raw.ch_names):
            return None

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            raw.pick_channels(desired, ordered=False)
            raw.load_data()
        
        raw.resample(SAMPLE_RATE, n_jobs=1)

        eeg_picks = [raw.ch_names.index(ch) for ch in EEG_EOG_ECG if ch in raw.ch_names]
        if eeg_picks:
            raw._data[eeg_picks] *= 1e6

        sao2_stats = {"n_imputed": 0, "frac_imputed": 0.0}
        if "SAO2" in raw.ch_names:
            idx = raw.ch_names.index("SAO2")
            clean_sig, stats = clean_sao2(raw._data[idx], SAMPLE_RATE)
            raw._data[idx] = clean_sig
            sao2_stats = stats

        meas_date = raw.info['meas_date']
        if meas_date is None: meas_date = datetime(2000, 1, 1, 0, 0, 0)
        if meas_date.tzinfo is not None: meas_date = meas_date.replace(tzinfo=None)

        valid_epochs = parse_txt_annotations(txt_path, meas_date)
        if not valid_epochs: return None
        current_map = {name: i for i, name in enumerate(raw.ch_names)}
        order_idx = [current_map[name] for name in desired]
        
        data_full = raw.get_data().astype(np.float32)
        del raw
        gc.collect()
        data_full = data_full[order_idx, :]
        
        epoch_list = []
        limit = data_full.shape[1]
        n_samp = int(EPOCH_SAMPLES)

        for onset, _ in valid_epochs:
            i0 = int(onset * SAMPLE_RATE)
            i1 = i0 + n_samp
            if i1 <= limit:
                epoch_list.append(data_full[:, i0:i1])

        if not epoch_list: return None

        epochs = np.stack(epoch_list, axis=0)

        np.savez_compressed(
            out_path,
            epochs=epochs,
            label=int(label_idx),
            label_str=label_str,
            filename=filename,
            sao2_stats=sao2_stats
        )
        return label_str

    except Exception as e:
        print(f"Error {filename}: {e}")
        return None

def run_preprocessing(force=False):
    existing = glob.glob(os.path.join(PROCESSED_DIR, "*.npz"))
    if existing and not force:
        print(f"Found {len(existing)} existing files skipping.")
        return

    files = sorted(glob.glob(os.path.join(RAW_EDF_DIR, "*.edf")))
    print(f"Found {len(files)} EDF files. Processing with n_jobs=4...")

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_single_file)(p) for p in files
    )

    n_ok = sum(1 for r in results if r is not None)
    print(f"\nDone. Processed {n_ok} files.")

if __name__ == "__main__":
    run_preprocessing(force=True)