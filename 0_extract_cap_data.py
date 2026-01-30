import numpy as np
import mne
import os
import argparse
import glob
import pandas as pd
import pickle
import sys
import gc
from datetime import datetime, timedelta
from loguru import logger
import warnings

import config
from config import PATH_TO_RAW_DATA, PATH_TO_PROCESSED_DATA

warnings.filterwarnings("ignore")

TARGET_SFREQ = 256

CAP_LABEL_MAP = {
    "SLEEP-S0": 0,    
    "SLEEP-S1": 1,    
    "SLEEP-S2": 2,    
    "SLEEP-S3": 3,    
    "SLEEP-S4": 3,    
    "SLEEP-REM": 4    
}

DROP_LABELS = ["SLEEP-UNSCORED", "MT", "Movement", "MCAP-A1", "MCAP-A2", "MCAP-A3"]

TARGET_CHANNELS_MAP = {
    "Respiratory": ["PLETH", "SaO2", "PLETH"],
    "Brain": ["F4-C4", "C4-A1", "C4-P4", "P4-O2", "ROC-LOC"],
    "ECG": ["ECG1-ECG2"]
}

def clean_sao2(x: np.ndarray, fs: float) -> np.ndarray:
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
    return x

def parse_txt_annotations(txt_path, edf_start_time):
    valid_epochs = []
    
    header_idx = -1
    if not os.path.exists(txt_path):
        return []

    with open(txt_path, 'r', encoding='latin-1') as f:
        for i, line in enumerate(f):
            if line.strip().startswith("Sleep Stage"):
                header_idx = i
                break
            
    if header_idx == -1:
        logger.warning(f"Could not find header in {txt_path}")
        return []

    try:
        df = pd.read_csv(txt_path, skiprows=header_idx, sep='\t', engine='python')
    except Exception as e:
        logger.error(f"Error reading CSV {txt_path}: {e}")
        return []
    
    df.columns = [c.strip() for c in df.columns]
    required_cols = ['Sleep Stage', 'Time [hh:mm:ss]', 'Event', 'Duration[s]']
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"Missing columns in {txt_path}. Found: {df.columns}")
        return []

    for _, row in df.iterrows():
        event = str(row['Event']).strip()
        try:
            duration = float(row['Duration[s]'])
        except ValueError:
            continue
            
        time_str = str(row['Time [hh:mm:ss]']).strip()
        
        if not event.startswith("SLEEP-"): continue
        if event in DROP_LABELS: continue
        if event not in CAP_LABEL_MAP: continue
        label_int = CAP_LABEL_MAP[event]
        
        if duration != 30.0: continue

        try:
            annot_time = datetime.strptime(time_str, '%H:%M:%S')
            onset_dt = edf_start_time.replace(hour=annot_time.hour, 
                                              minute=annot_time.minute, 
                                              second=annot_time.second)
            if onset_dt < edf_start_time:
                onset_dt += timedelta(days=1)
            onset_sec = (onset_dt - edf_start_time).total_seconds()
            
            if onset_sec < 0: continue
            valid_epochs.append((onset_sec, label_int))
        except Exception as e:
            continue

    return valid_epochs

def get_channel_data(raw, target_names, n_samples_expected, scale_microvolts=False):
    data_list = []
    for target in target_names:
        found_ch = None
        for ch in raw.ch_names:
            if ch.lower() == target.lower():
                found_ch = ch
                break
        
        if found_ch:
            d = raw.get_data(picks=found_ch).flatten()
            if scale_microvolts:
                d *= 1e6

            if len(d) > n_samples_expected:
                d = d[:n_samples_expected]
            elif len(d) < n_samples_expected:
                d = np.pad(d, (0, n_samples_expected - len(d)), 'constant')
        else:
            d = np.zeros(n_samples_expected, dtype=np.float32)
        data_list.append(d)
        
    return np.vstack(data_list)

def process_patient_file(edf_path, txt_path, output_dir):
    file_prefix = os.path.basename(edf_path).split('.')[0]
    path_to_patient_X = os.path.join(output_dir, "X", file_prefix)
    path_to_patient_Y = os.path.join(output_dir, "Y", f"{file_prefix}.pickle")

    if os.path.exists(path_to_patient_Y):
        logger.info(f"Skipping {file_prefix}, already processed.")
        return

    logger.info(f"Processing {file_prefix}...")

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        logger.error(f"Failed to load EDF {edf_path}: {e}")
        return

    if raw.info['sfreq'] != TARGET_SFREQ:
        try:

            raw.resample(TARGET_SFREQ, npad="auto", n_jobs=4, verbose=False)
        except Exception as e:
            logger.error(f"Resampling failed for {file_prefix}: {e}")
            return

    meas_date = raw.info['meas_date']
    if meas_date is None:
        logger.warning(f"No measurement date in header for {file_prefix}. using dummy.")
        meas_date = datetime(2000, 1, 1, 0, 0, 0)
    
    if meas_date.tzinfo is not None:
        meas_date = meas_date.replace(tzinfo=None)

    valid_epochs = parse_txt_annotations(txt_path, meas_date)
    
    if not valid_epochs:
        logger.warning(f"No valid epochs found for {file_prefix}")
        del raw
        gc.collect()
        return

    os.makedirs(path_to_patient_X, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Y"), exist_ok=True)
    
    labels_dict = {}
    n_samples_epoch = int(30 * TARGET_SFREQ)
    total_samples = raw.n_times


    resp_full = get_channel_data(raw, TARGET_CHANNELS_MAP["Respiratory"], total_samples, scale_microvolts=False)
    

    brain_full = get_channel_data(raw, TARGET_CHANNELS_MAP["Brain"], total_samples, scale_microvolts=True)
    
    ecg_full = get_channel_data(raw, TARGET_CHANNELS_MAP["ECG"], total_samples, scale_microvolts=True)

    if "SaO2" in TARGET_CHANNELS_MAP["Respiratory"]:
        sao2_idx = TARGET_CHANNELS_MAP["Respiratory"].index("SaO2")
        resp_full[sao2_idx, :] = clean_sao2(resp_full[sao2_idx, :], TARGET_SFREQ)

    del raw
    gc.collect()

    count_saved = 0
    for idx, (onset_sec, label_int) in enumerate(valid_epochs):
        start_sample = int(onset_sec * TARGET_SFREQ)
        end_sample = start_sample + n_samples_epoch
        
        if end_sample > total_samples:
            continue
            
        resp_epoch = resp_full[:, start_sample:end_sample]
        brain_epoch = brain_full[:, start_sample:end_sample]
        ecg_epoch = ecg_full[:, start_sample:end_sample]
        
        final_tensor = np.vstack([resp_epoch, brain_epoch, ecg_epoch])
        
        filename = f"{file_prefix}_{idx}.npy"
        np.save(os.path.join(path_to_patient_X, filename), final_tensor.astype(np.float32))
        
        labels_dict[filename] = label_int
        count_saved += 1

    with open(path_to_patient_Y, 'wb') as f:
        pickle.dump(labels_dict, f)
        
    logger.info(f"Saved {count_saved} epochs for {file_prefix}")
    
    del resp_full, brain_full, ecg_full
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Process CAP Sleep Database for SleepFM")
    parser.add_argument("--data_path", type=str, default=None, help="Path to Raw Data")
    parser.add_argument("--save_path", type=str, default=None, help="Path to Processed Data")
    args = parser.parse_args()

    input_dir = args.data_path if args.data_path else PATH_TO_RAW_DATA
    output_dir = args.save_path if args.save_path else PATH_TO_PROCESSED_DATA

    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "X"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Y"), exist_ok=True)

    edf_files = glob.glob(os.path.join(input_dir, "*.edf"))
    logger.info(f"Found {len(edf_files)} EDF files in {input_dir}")
    
    for edf_path in edf_files:
        txt_path = edf_path.replace(".edf", ".txt")
        if os.path.exists(txt_path):
            process_patient_file(edf_path, txt_path, output_dir)
        else:
            logger.warning(f"Skipping {os.path.basename(edf_path)}: no matching .txt annotation file found.")

if __name__ == "__main__":
    main()