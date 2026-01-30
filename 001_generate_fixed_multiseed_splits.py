import os
import json
import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from collections import Counter

DISORDER_MAP = {
    'n': 0,      
    'nfle': 1,  
    'plm': 2,    
    'ins': 3,    
    'rbd': 4,   
}

ID_TO_NAME = {
    0: "Control",
    1: "NFLE",
    2: "PLM",
    3: "Insomnia",
    4: "RBD"
}

SEEDS = range(42, 92)
N_FOLDS = 4

def get_prefix(subject_id):
    return "".join([c for c in subject_id if not c.isdigit()])

def load_baseline_subjects(csv_path):

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Baseline CSV not found at: {csv_path}")
        
    print(f"Loading Baseline CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'npz_file' not in df.columns:
        raise ValueError("CSV missing 'npz_file' column")
        
    subjects = df['npz_file'].apply(lambda x: str(x).replace('.npz', '')).unique().tolist()
    print(f"  -> Found {len(subjects)} unique subjects in Baseline CSV.")
    return set(subjects)

def load_sleepfm_subjects(dataset_path):

    if not dataset_path or not os.path.exists(dataset_path):
        print("SleepFM dataset path not provided or not found. Skipping intersection check.")
        return None

    subjects = set()
    
    pickle_path = os.path.join(dataset_path, "dataset_events.pickle") 
    pickle_map_path = os.path.join(dataset_path, "dataset.pickle")

    if os.path.exists(pickle_map_path):
        print(f"Loading SleepFM subjects from: {pickle_map_path}")
        with open(pickle_map_path, 'rb') as f:
            data = pickle.load(f)
            for split, split_data in data.items():
                for item in split_data:
                    subject_id = list(item.keys())[0]
                    subjects.add(subject_id)
    
    else:
        print(f"Scanning SleepFM directory: {dataset_path}")
        x_dir = os.path.join(dataset_path, "X")
        if os.path.exists(x_dir):
            for subj in os.listdir(x_dir):
                if os.path.isdir(os.path.join(x_dir, subj)):
                    subjects.add(subj)
        else:
            print("Could not find 'dataset.pickle' or 'X' directory. Cannot verify SleepFM subjects.")
            return None

    print(f"Found {len(subjects)} unique subjects in SleepFM dataset.")
    return subjects

def main():
    parser = argparse.ArgumentParser(description="Generate Golden Splits for Scientific Comparison (Multi-Seed)")
    parser.add_argument("--baseline_csv", type=str, 
                        default="/content/baseline_codebase_sleep_night_features/sleep_night_features_8.csv",
                        help="Path to the Baseline Features CSV")
    parser.add_argument("--sleepfm_dir", type=str, default="/content/processed",
                        help="Path to SleepFM processed data directory (optional but recommended for intersection)")
    parser.add_argument("--output_file", type=str, default="common_splits_multiseed.json",
                        help="Output JSON file for splits")
    
    args = parser.parse_args()
    
    baseline_ids = load_baseline_subjects(args.baseline_csv)
    sleepfm_ids = load_sleepfm_subjects(args.sleepfm_dir)
    
    if sleepfm_ids is not None:
        common_ids = list(baseline_ids.intersection(sleepfm_ids))
        print(f"Intersection:{len(common_ids)} subjects common to both pipelines.")
        
        missing_in_sleepfm = baseline_ids - sleepfm_ids
        if missing_in_sleepfm:
            print(f"{len(missing_in_sleepfm)} subjects in Baseline but missing in SleepFM (will be dropped).")
    else:
        print("Using all Baseline subjects (SleepFM verification skipped).")
        common_ids = list(baseline_ids)

    final_subjects = []
    final_labels = []
    skipped_counts = Counter()

    for pid in common_ids:
        prefix = get_prefix(pid)
        
        if prefix in DISORDER_MAP:
            label = DISORDER_MAP[prefix]
            final_subjects.append(pid)
            final_labels.append(label)
        else:
            skipped_counts[prefix] += 1
    
    print("\nFiltering for Target Classes (Control, NFLE, PLM, Insomnia, RBD)")
    print(f"Kept: {len(final_subjects)} subjects.")
    print(f"Dropped (Excluded/Unknown prefixes): {dict(skipped_counts)}")

    if len(final_subjects) < N_FOLDS:
        raise ValueError(f"Not enough subjects ({len(final_subjects)}) for {N_FOLDS}-fold splitting!")

    splits_output_dict = {}

    X_dummy = np.zeros(len(final_subjects))
    y = np.array(final_labels)
    subj_array = np.array(final_subjects)

    print(f"\nGenerating {N_FOLDS} Fold Stratified Splits over {len(SEEDS)} Seeds ({SEEDS[0]}-{SEEDS[-1]})")
    
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        seed_folds = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_dummy, y)):
            train_subjs = subj_array[train_idx].tolist()
            test_subjs = subj_array[test_idx].tolist()
            
            seed_folds.append({
                "fold_idx": fold_idx,
                "train": train_subjs,
                "test": test_subjs
            })
        
        splits_output_dict[f"seed_{seed}"] = seed_folds

    with open(args.output_file, 'w') as f:
        json.dump(splits_output_dict, f, indent=2)

if __name__ == "__main__":
    main()