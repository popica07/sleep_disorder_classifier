import os
import json
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize

BASE_DIR = "/content/baseline_codebase_sleep_night_features"
CSV_PATH = os.path.join(BASE_DIR, "sleep_night_features_8.csv")
SPLITS_JSON_PATH = "common_splits_multiseed.json"
OUT_DIR = os.path.join(BASE_DIR, "night_classification_results", "baseline")
CM_OUT_DIR = "/content/final-aligned-results/Statistical_Features_Baseline_cms"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CM_OUT_DIR, exist_ok=True)

TARGET_K_FOR_COMPARISON = 5 

SCENARIOS = [
    {"name": "A_UniformWeights",            "weights": "uniform",  "oversample_noise": False},
    {"name": "C_DistanceWeights",           "weights": "distance", "oversample_noise": False},
]

OVERSAMPLE_LEVELS = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]

for lvl in OVERSAMPLE_LEVELS:
    pct = int(lvl * 100)
    

    SCENARIOS.append({
        "name": f"B_Uniform_F_{pct}_oversample",
        "weights": "uniform",
        "oversample_noise": True,
        "noise_mode": "fixed",
        "noise_level": lvl
    })

    SCENARIOS.append({
        "name": f"B_Uniform_A_{pct}_oversample",
        "weights": "uniform",
        "oversample_noise": True,
        "noise_mode": "adaptive",
        "noise_level": lvl
    })


for lvl in OVERSAMPLE_LEVELS:
    pct = int(lvl * 100)
    

    SCENARIOS.append({
        "name": f"D_Distance_F_{pct}_oversample",
        "weights": "distance",
        "oversample_noise": True,
        "noise_mode": "fixed",
        "noise_level": lvl
    })
    
    SCENARIOS.append({
        "name": f"D_Distance_A_{pct}_oversample",
        "weights": "distance",
        "oversample_noise": True,
        "noise_mode": "adaptive",
        "noise_level": lvl
    })


STR_TO_INT_MAP = {
    'healthy_control': 0, 'nocturnal_frontal_lobe_epilepsy': 1,
    'periodic_leg_movements': 2, 'insomnia': 3, 'rem_behavior_disorder': 4
}
DISORDER_NAMES = ["Control", "NFLE", "PLM", "Insomnia", "RBD"]

def calculate_probabilistic_metrics(y_true, y_probs, n_classes=5):

    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    auroc_list, auprc_list = [], []
    
    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) == 0: continue
        auroc_list.append(roc_auc_score(y_true_bin[:, i], y_probs[:, i]))
        auprc_list.append(average_precision_score(y_true_bin[:, i], y_probs[:, i]))
        
    return np.mean(auroc_list) if auroc_list else 0.0, np.mean(auprc_list) if auprc_list else 0.0

def make_pipeline(k, weights_param):
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights=weights_param, n_jobs=-1))
    ])

def get_augmented_train_data(X_train, y_train, noise_mode='fixed', noise_level=0.01):
    counts = Counter(y_train)
    mean_count = np.mean(list(counts.values()))
    minority_classes = [cls for cls, count in counts.items() if count < mean_count]
    
    X_augmented_list = [X_train]
    y_augmented_list = [y_train]
    
    for cls in minority_classes:
        cls_indices = np.where(y_train == cls)[0]
        if len(cls_indices) > 0:
            X_cls = X_train[cls_indices]
            
            if noise_mode == 'adaptive':
                feature_stds = np.std(X_cls, axis=0)
                feature_stds[feature_stds == 0] = 0 
                noise = np.random.normal(0, 1, X_cls.shape) * feature_stds * noise_level
            else:
                noise = np.random.normal(0, noise_level, X_cls.shape)
            
            X_augmented_list.append(X_cls + noise)
            y_augmented_list.append(y_train[cls_indices])
    
    return np.vstack(X_augmented_list), np.concatenate(y_augmented_list)

def save_pooled_confusion_matrix(y_true, y_pred, scenario_name, model_name="KNN"):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=DISORDER_NAMES, yticklabels=DISORDER_NAMES)
    plt.title(f'Pooled CM: {model_name} - {scenario_name} (Normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filename = f"CM_{model_name}_{scenario_name}.png"
    out_path = os.path.join(CM_OUT_DIR, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved Pooled CM to {out_path}")

def main():
    if not os.path.exists(CSV_PATH): raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    if not os.path.exists(SPLITS_JSON_PATH): raise FileNotFoundError(f"Splits JSON not found: {SPLITS_JSON_PATH}")

    df = pd.read_csv(CSV_PATH)
    with open(SPLITS_JSON_PATH, 'r') as f:
        splits_dict = json.load(f)
    df['subject_id'] = df['npz_file'].astype(str).str.replace('.npz', '', regex=False)
    
    all_valid_subjects = set()
    first_seed_folds = list(splits_dict.values())[0]
    for fold in first_seed_folds:
        all_valid_subjects.update(fold['train'])
        all_valid_subjects.update(fold['test'])
        
    df_filtered = df[df['subject_id'].isin(all_valid_subjects)].copy()
    df_filtered['target'] = df_filtered['label_str'].map(STR_TO_INT_MAP)
    df_filtered = df_filtered.dropna(subset=['target'])
    df_filtered['target'] = df_filtered['target'].astype(int)
    
    feature_cols = [c for c in df.columns if c not in ['label', 'label_str', 'npz_file', 'edf_filename', 'subject_id', 'target']]
    
    print(f"Data Loaded: {df_filtered.shape} | Seeds: {len(splits_dict)}")

    for scenario in SCENARIOS:
        sc_name = scenario["name"]
        weights_config = scenario["weights"]
        print(f"\n--- Processing KNN Scenario: {sc_name} (K={TARGET_K_FOR_COMPARISON}, weights={weights_config}) ---")
        
        seed_metrics = {}
        grand_y_true = []
        grand_y_pred = []
        
        for seed_key, folds in tqdm(splits_dict.items(), desc=f"KNN {sc_name}"):
            
            fold_f1s = []
            fold_aurocs = []
            fold_auprcs = []
            
            for fold_data in folds:
                train_mask = df_filtered['subject_id'].isin(fold_data['train'])
                test_mask = df_filtered['subject_id'].isin(fold_data['test'])
                
                X_train = df_filtered.loc[train_mask, feature_cols].values
                y_train = df_filtered.loc[train_mask, 'target'].values
                X_test = df_filtered.loc[test_mask, feature_cols].values
                y_test = df_filtered.loc[test_mask, 'target'].values
                
                train_means = np.nanmean(X_train, axis=0)
                train_means[np.isnan(train_means)] = 0.0
                
                inds_train = np.where(np.isnan(X_train))
                X_train[inds_train] = np.take(train_means, inds_train[1])
                
                inds_test = np.where(np.isnan(X_test))
                X_test[inds_test] = np.take(train_means, inds_test[1])

                if scenario["oversample_noise"]:
                    mode = scenario.get("noise_mode", "fixed")
                    level = scenario.get("noise_level", 0.01)
                    X_train, y_train = get_augmented_train_data(X_train, y_train, noise_mode=mode, noise_level=level)
                
                model = make_pipeline(TARGET_K_FOR_COMPARISON, weights_config)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                y_probs = model.predict_proba(X_test)

                f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
                auroc, auprc = calculate_probabilistic_metrics(y_test, y_probs)
                
                fold_f1s.append(f1)
                fold_aurocs.append(auroc)
                fold_auprcs.append(auprc)
                
                grand_y_true.extend(y_test)
                grand_y_pred.extend(y_pred)
            
            seed_metrics[seed_key] = {
                "f1": np.mean(fold_f1s),
                "auroc": np.mean(fold_aurocs),
                "auprc": np.mean(fold_auprcs)
            }

        full_report = classification_report(
            grand_y_true, grand_y_pred, target_names=DISORDER_NAMES, digits=3, zero_division=0
        )
        
        results_pkg = {
            'scenario': sc_name,
            'model': 'knn',
            'config': scenario,
            'seed_metrics': seed_metrics,
            'y_true_all': grand_y_true,
            'y_pred_all': grand_y_pred,
            'report': full_report
        }
        
        filename = f"results_knn_{sc_name}_multiseed.pkl"
        out_path = os.path.join(OUT_DIR, filename)
        with open(out_path, 'wb') as f:
            pickle.dump(results_pkg, f)
            
        print(f"Saved {sc_name} to {filename}")

        save_pooled_confusion_matrix(grand_y_true, grand_y_pred, sc_name, model_name="KNN")

if __name__ == "__main__":
    main()