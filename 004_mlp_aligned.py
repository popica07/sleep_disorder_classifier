import os
import json
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, 
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR = "/content/baseline_codebase_sleep_night_features"
CSV_PATH = os.path.join(BASE_DIR, "sleep_night_features_8.csv")
SPLITS_JSON_PATH = "common_splits_multiseed.json"
OUT_DIR = os.path.join(BASE_DIR, "night_classification_results", "baseline")
CM_OUT_DIR = "/content/final-aligned-results/Statistical_Features_Baseline_cms"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CM_OUT_DIR, exist_ok=True)

MAX_EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-3
HIDDEN_DIM = 128
PATIENCE = 15
MIN_IMPROVEMENT = 1e-4

STR_TO_INT_MAP = {
    'healthy_control': 0, 'nocturnal_frontal_lobe_epilepsy': 1,
    'periodic_leg_movements': 2, 'insomnia': 3, 'rem_behavior_disorder': 4
}
DISORDER_NAMES = ["Control", "NFLE", "PLM", "Insomnia", "RBD"]
N_CLASSES = 5

SCENARIOS = [
    {"name": "A_None",         "weights": False, "focal": False}, 
    {"name": "B_FocalWeight",  "weights": False, "focal": True}, 
    {"name": "C_WeightsOnly",  "weights": True,  "focal": False},  
]

OVERSAMPLE_LEVELS = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]

for lvl in OVERSAMPLE_LEVELS:
    pct = int(lvl * 100)
    SCENARIOS.append({
        "name": f"D_F_{pct}_oversample",
        "weights": True,
        "focal": False,
        "oversample_noise": True,
        "noise_mode": "fixed",
        "noise_level": lvl
    })

for lvl in OVERSAMPLE_LEVELS:
    pct = int(lvl * 100)
    SCENARIOS.append({
        "name": f"D_A_{pct}_oversample",
        "weights": True,
        "focal": False,
        "oversample_noise": True,
        "noise_mode": "adaptive",
        "noise_level": lvl
    })

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha 

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def calculate_probabilistic_metrics(y_true, y_probs, n_classes=5):
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    auroc_list, auprc_list = [], []
    
    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) == 0: continue
        auroc_list.append(roc_auc_score(y_true_bin[:, i], y_probs[:, i]))
        auprc_list.append(average_precision_score(y_true_bin[:, i], y_probs[:, i]))
        
    return np.mean(auroc_list) if auroc_list else 0.0, np.mean(auprc_list) if auprc_list else 0.0

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

def save_pooled_confusion_matrix(y_true, y_pred, scenario_name, model_name="MLP"):
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
        
    print(f"Loaded CSV: {df.shape}")
    print(f"Loaded Splits: {len(splits_dict)} seeds found.")

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

    for scenario in SCENARIOS:
        sc_name = scenario["name"]
        print(f"\n--- Processing MLP Scenario: {sc_name} ---")
        
        seed_metrics = {}
        grand_y_true = []
        grand_y_pred = []
        
        for seed_key, folds in tqdm(splits_dict.items(), desc=f"MLP {sc_name}"):
            seed_int = int(seed_key.replace("seed_", ""))
            
            fold_f1s = []
            fold_aurocs = []
            fold_auprcs = []
            
            for fold_data in folds:
                torch.manual_seed(seed_int)
                np.random.seed(seed_int)

                train_mask = df_filtered['subject_id'].isin(fold_data['train'])
                test_mask = df_filtered['subject_id'].isin(fold_data['test'])
                
                X_train = df_filtered.loc[train_mask, feature_cols].values.astype(np.float32)
                y_train = df_filtered.loc[train_mask, 'target'].values.astype(int)
                X_test = df_filtered.loc[test_mask, feature_cols].values.astype(np.float32)
                y_test = df_filtered.loc[test_mask, 'target'].values.astype(int)
                
                train_means = np.nanmean(X_train, axis=0)
                train_means[np.isnan(train_means)] = 0.0 
                
                inds_nan_train = np.where(np.isnan(X_train))
                X_train[inds_nan_train] = np.take(train_means, inds_nan_train[1])
                
                inds_nan_test = np.where(np.isnan(X_test))
                X_test[inds_nan_test] = np.take(train_means, inds_nan_test[1])

                if scenario.get("oversample_noise", False):
                    mode = scenario.get("noise_mode", "fixed")
                    level = scenario.get("noise_level", 0.01)
                    X_train, y_train = get_augmented_train_data(X_train, y_train, noise_mode=mode, noise_level=level)

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                X_train_t = torch.tensor(X_train, dtype=torch.float32)
                y_train_t = torch.tensor(y_train, dtype=torch.long)
                X_test_t = torch.tensor(X_test, dtype=torch.float32)
                
                train_loader = DataLoader(
                    TensorDataset(X_train_t, y_train_t),
                    batch_size=BATCH_SIZE,
                    shuffle=True
                )

                criterion = None
                cw_tensor = None
                
                if scenario.get("weights", False) or scenario.get("focal", False):
                    classes = np.unique(y_train)
                    weights_calc = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
                    
                    full_weights = np.ones(N_CLASSES, dtype=np.float32)
                    for cls, w in zip(classes, weights_calc):
                        full_weights[cls] = w
                    
                    if scenario.get("focal", False):
                        full_weights = np.square(full_weights)

                    cw_tensor = torch.tensor(full_weights, dtype=torch.float32)
                
                if scenario.get("focal", False):
                    criterion = FocalLoss(alpha=cw_tensor, gamma=2.0)
                elif scenario.get("weights", False):
                    criterion = nn.CrossEntropyLoss(weight=cw_tensor)
                else:
                    criterion = nn.CrossEntropyLoss()

                model = MLP(input_dim=X_train.shape[1], hidden_dim=HIDDEN_DIM, num_classes=N_CLASSES)
                optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                
                model.train()
                best_loss = float("inf")
                bad_epochs = 0
                
                for epoch in range(MAX_EPOCHS):
                    epoch_losses = []
                    for xb, yb in train_loader:
                        optimizer.zero_grad()
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        loss.backward()
                        optimizer.step()
                        epoch_losses.append(loss.item())
                    
                    mean_loss = np.mean(epoch_losses) if epoch_losses else 0.0
                    
                    if mean_loss < best_loss - MIN_IMPROVEMENT:
                        best_loss = mean_loss
                        bad_epochs = 0
                    else:
                        bad_epochs += 1
                    
                    if bad_epochs >= PATIENCE:
                        break
                
                model.eval()
                with torch.no_grad():
                    logits = model(X_test_t)
                    y_probs = torch.softmax(logits, dim=1).cpu().numpy()
                    y_pred = torch.argmax(logits, dim=1).cpu().numpy()
                
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
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
            'model': 'mlp',
            'config': scenario,
            'seed_metrics': seed_metrics,
            'y_true_all': grand_y_true,
            'y_pred_all': grand_y_pred,
            'report': full_report
        }
        
        filename = f"results_mlp_{sc_name}_multiseed.pkl"
        out_path = os.path.join(OUT_DIR, filename)
        with open(out_path, 'wb') as f:
            pickle.dump(results_pkg, f)
            
        print(f"Saved {sc_name} to {filename}")

        save_pooled_confusion_matrix(grand_y_true, grand_y_pred, sc_name, model_name="MLP")

if __name__ == "__main__":
    main()