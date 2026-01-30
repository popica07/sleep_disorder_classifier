import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from loguru import logger
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, 
    f1_score, 
    confusion_matrix, 
    accuracy_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight

import config
from config import DISORDER_NAMES, PATH_TO_PROCESSED_DATA
import utils

MLP_MAX_EPOCHS = 20
MLP_BATCH_SIZE = 16
MLP_LR = 1e-3
MLP_HIDDEN_DIM = 128
MLP_PATIENCE = 15
MLP_MIN_IMPROVEMENT = 1e-4

CM_OUT_DIR = "/content/final-aligned-results/SleepFM_Embeddings_cms"
os.makedirs(CM_OUT_DIR, exist_ok=True)


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
    
    auroc_list = []
    auprc_list = []
    
    for i in range(n_classes):
        if i >= y_probs.shape[1]:
            continue

        if np.sum(y_true_bin[:, i]) == 0:
            continue
            
        try:
            auroc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
            auprc = average_precision_score(y_true_bin[:, i], y_probs[:, i])
            auroc_list.append(auroc)
            auprc_list.append(auprc)
        except Exception as e:
            continue
        
    return np.mean(auroc_list) if auroc_list else 0.0, np.mean(auprc_list) if auprc_list else 0.0

def aggregate_features(patient_data, method="mean_std"):

    X_agg = []
    patient_ids_ordered = []
    sorted_pids = sorted(patient_data.keys())
    
    for pid in sorted_pids:
        tensor = patient_data[pid] 
        if method == "mean_std":
            mean_vec = torch.mean(tensor, dim=0).numpy()
            std_vec = torch.std(tensor, dim=0).numpy()
            feat_vec = np.concatenate([mean_vec, std_vec])
        elif method == "mean":
            feat_vec = torch.mean(tensor, dim=0).numpy()
        X_agg.append(feat_vec)
        patient_ids_ordered.append(pid)
        
    return np.array(X_agg), patient_ids_ordered

def get_golden_fold_indices(splits_dict, ordered_ids):
    id_to_idx = {pid: i for i, pid in enumerate(ordered_ids)}
    
    all_seeds_folds = {}
    
    for seed_key, folds_list in splits_dict.items():
        parsed_folds = []
        for fold_item in folds_list:
            train_subjs = fold_item['train']
            test_subjs = fold_item['test']
            
            train_idx = [id_to_idx[s] for s in train_subjs if s in id_to_idx]
            test_idx = [id_to_idx[s] for s in test_subjs if s in id_to_idx]
            
            parsed_folds.append((np.array(train_idx), np.array(test_idx)))
        all_seeds_folds[seed_key] = parsed_folds
        
    return all_seeds_folds

def save_pooled_confusion_matrix(y_true, y_pred, scenario_name, model_name):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=DISORDER_NAMES, yticklabels=DISORDER_NAMES)
    plt.title(f'Pooled CM: {model_name.upper()} - {scenario_name} (Normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filename = f"CM_{model_name}_{scenario_name}.png"
    out_path = os.path.join(CM_OUT_DIR, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved Pooled CM to {out_path}")

def apply_oversampling(X_train, y_train, noise_mode='fixed', noise_level=0.01):
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

def train_baseline_scenario(model_type, X_matrix, y_vector, ordered_ids, patient_labels, output_dir, scenario_config, golden_folds_dict=None):
    scenario_name = scenario_config["name"]
    use_weights = scenario_config.get("weights", False) 
    use_focal = scenario_config.get("focal", False)
    use_noise = scenario_config.get("oversample_noise", False)
    
    if golden_folds_dict is not None:
        seed_iterator = golden_folds_dict.items()
    else:
        logger.warning("No Golden Splits provided. Running single seed 42 generation.")
        folds = utils.get_cv_splits(ordered_ids, patient_labels, n_splits=5, stratified=True)
        seed_iterator = [("seed_42", folds)]
    
    logger.info(f"SCENARIO: {model_type.upper()} | {scenario_name} | Multi-Seed Eval")

    seed_metrics = {}
    accumulated_y_true = []
    accumulated_y_pred = []
    oversampling_debug = [] 
    
    for seed_key, folds in tqdm(seed_iterator, desc=f"{model_type} {scenario_name}", leave=False):
        seed_int = int(seed_key.replace("seed_", ""))
        torch.manual_seed(seed_int)
        np.random.seed(seed_int)
        
        fold_f1s = []
        fold_aurocs = []
        fold_auprcs = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X_matrix[train_idx], X_matrix[test_idx]
            y_train, y_test = y_vector[train_idx], y_vector[test_idx]

            train_means = np.nanmean(X_train, axis=0)
            train_means[np.isnan(train_means)] = 0.0
            
            inds_nan_train = np.where(np.isnan(X_train))
            X_train[inds_nan_train] = np.take(train_means, inds_nan_train[1])
            
            inds_nan_test = np.where(np.isnan(X_test))
            X_test[inds_nan_test] = np.take(train_means, inds_nan_test[1])

            if use_noise:
                mode = scenario_config.get("noise_mode", "fixed")
                level = scenario_config.get("noise_level", 0.01)

                if len(oversampling_debug) < len(folds):
                    counts = Counter(y_train)
                    mean_count = np.mean(list(counts.values()))
                    oversampling_debug.append({
                        "fold": fold_idx,
                        "original_counts": dict(counts),
                        "threshold": mean_count,
                        "mode": mode,
                        "level": level
                    })
                
                X_train, y_train = apply_oversampling(X_train, y_train, noise_mode=mode, noise_level=level)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            class_weight_dict = None
            cw_tensor = None
            
            if (isinstance(use_weights, bool) and use_weights) or use_focal:
                classes = np.unique(y_train)
                weights_calc = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
                
                full_weights = np.ones(5, dtype=np.float32)
                for cls, w in zip(classes, weights_calc):
                    full_weights[cls] = w

                if use_focal:
                    full_weights = np.square(full_weights)

                class_weight_dict = dict(zip(range(5), full_weights))
                cw_tensor = torch.tensor(full_weights, dtype=torch.float32)

            clf = None
            y_pred = None
            y_probs = None

            if model_type == "logistic":
                clf = LogisticRegression(class_weight=class_weight_dict, max_iter=2000, solver='liblinear', penalty='l1', C=0.1, random_state=seed_int)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_probs = clf.predict_proba(X_test)
                
            elif model_type == "rf":
                clf = RandomForestClassifier(n_estimators=100, class_weight=class_weight_dict, random_state=seed_int, max_depth=None)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_probs = clf.predict_proba(X_test)
                
            elif model_type == "knn":
                w_param = use_weights if isinstance(use_weights, str) else 'uniform'
                clf = KNeighborsClassifier(n_neighbors=5, weights=w_param, n_jobs=-1)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_probs = clf.predict_proba(X_test)

            elif model_type == "mlp":
                X_train_t = torch.tensor(X_train, dtype=torch.float32)
                y_train_t = torch.tensor(y_train, dtype=torch.long)
                X_test_t = torch.tensor(X_test, dtype=torch.float32)
                
                train_loader = DataLoader(
                    TensorDataset(X_train_t, y_train_t),
                    batch_size=MLP_BATCH_SIZE,
                    shuffle=True
                )
                
                model = MLP(input_dim=X_train.shape[1], hidden_dim=MLP_HIDDEN_DIM, num_classes=5)
                optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR)
                
                if use_focal:
                    criterion = FocalLoss(alpha=cw_tensor, gamma=2.0)
                elif isinstance(use_weights, bool) and use_weights:
                    criterion = nn.CrossEntropyLoss(weight=cw_tensor)
                else:
                    criterion = nn.CrossEntropyLoss()
                
                model.train()
                best_loss = float("inf")
                bad_epochs = 0
                
                for epoch in range(MLP_MAX_EPOCHS):
                    epoch_losses = []
                    for xb, yb in train_loader:
                        optimizer.zero_grad()
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        loss.backward()
                        optimizer.step()
                        epoch_losses.append(loss.item())
                    
                    mean_loss = np.mean(epoch_losses) if epoch_losses else 0.0
                    
                    if mean_loss < best_loss - MLP_MIN_IMPROVEMENT:
                        best_loss = mean_loss
                        bad_epochs = 0
                    else:
                        bad_epochs += 1
                    
                    if bad_epochs >= MLP_PATIENCE:
                        break
                
                model.eval()
                with torch.no_grad():
                    logits = model(X_test_t)
                    y_probs_t = torch.softmax(logits, dim=1)
                    y_pred = torch.argmax(logits, dim=1).cpu().numpy()
                    y_probs = y_probs_t.cpu().numpy()

            fold_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            auroc, auprc = calculate_probabilistic_metrics(y_test, y_probs, n_classes=5)
            
            fold_f1s.append(fold_f1)
            fold_aurocs.append(auroc)
            fold_auprcs.append(auprc)
            
            accumulated_y_true.extend(y_test)
            accumulated_y_pred.extend(y_pred)
            
        seed_metrics[seed_key] = {
            "f1": np.mean(fold_f1s),
            "auroc": np.mean(fold_aurocs),
            "auprc": np.mean(fold_auprcs)
        }

    avg_f1_all_seeds = np.mean([m['f1'] for m in seed_metrics.values()])
    
    full_report = classification_report(
        accumulated_y_true, accumulated_y_pred, target_names=DISORDER_NAMES, digits=3, zero_division=0
    )
    
    logger.info(f"Finished {model_type.upper()} - {scenario_name}. Avg Macro F1 (Across Seeds): {avg_f1_all_seeds:.3f}")

    results_pkg = {
        'scenario': scenario_name,
        'model': model_type,
        'config': scenario_config,
        'seed_metrics': seed_metrics, 
        'y_true_all': accumulated_y_true, 
        'y_pred_all': accumulated_y_pred,
        'report': full_report,
        'oversampling_debug': oversampling_debug 
    }
    
    filename = f"results_{model_type}_{scenario_name}_multiseed.pkl"
    with open(os.path.join(output_dir, filename), 'wb') as f:
        pickle.dump(results_pkg, f)

    save_pooled_confusion_matrix(accumulated_y_true, accumulated_y_pred, scenario_name, model_type)

def run_baseline_experiment(args):
    if args.dataset_dir is None:
        args.dataset_dir = PATH_TO_PROCESSED_DATA
        
    output_dir = os.path.join(args.dataset_dir, "night_classification_results", "baseline")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.add(os.path.join(output_dir, "baseline_training.log"))
    logger.info("Loading and grouping nights...")
    
    patient_data, patient_labels, valid_ids = utils.load_and_group_nights(args.dataset_dir)
    
    golden_folds_dict = None
    if args.use_golden_splits:
        logger.info(f"Loading Golden Splits from: {args.use_golden_splits}")
        with open(args.use_golden_splits, 'r') as f:
            splits_raw_dict = json.load(f)
        valid_golden_ids = set()
        first_seed = list(splits_raw_dict.values())[0]
        for fold in first_seed:
            valid_golden_ids.update(fold['train'])
            valid_golden_ids.update(fold['test'])
            
        valid_ids = [pid for pid in valid_ids if pid in valid_golden_ids]
        patient_data = {pid: patient_data[pid] for pid in valid_ids}
        patient_labels = {pid: patient_labels[pid] for pid in valid_ids}
        
        if len(valid_ids) == 0:
            raise ValueError("Golden Split filtering removed ALL patients!")

    X_matrix, ordered_ids = aggregate_features(patient_data, method=args.agg_method)
    y_vector = np.array([patient_labels[pid] for pid in ordered_ids])
    
    if args.use_golden_splits:
        golden_folds_dict = get_golden_fold_indices(splits_raw_dict, ordered_ids)

    OVERSAMPLE_LEVELS = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]

    SCENARIOS_GENERIC = [
        {"name": "A_None",         "weights": False, "focal": False}, 
        {"name": "B_FocalWeight",  "weights": False, "focal": True},
        {"name": "C_WeightsOnly",  "weights": True,  "focal": False},  
    ]
    

    for lvl in OVERSAMPLE_LEVELS:
        pct = int(lvl * 100)
        SCENARIOS_GENERIC.append({
            "name": f"D_F_{pct}_oversample",
            "weights": True, 
            "focal": False, 
            "oversample_noise": True, 
            "noise_mode": "fixed", 
            "noise_level": lvl
        })
        SCENARIOS_GENERIC.append({
            "name": f"D_A_{pct}_oversample",
            "weights": True, 
            "focal": False, 
            "oversample_noise": True, 
            "noise_mode": "adaptive", 
            "noise_level": lvl
        })

    SCENARIOS_KNN = [
        {"name": "A_UniformWeights",  "weights": "uniform",  "oversample_noise": False},
        {"name": "C_DistanceWeights", "weights": "distance", "oversample_noise": False},
    ]

    for lvl in OVERSAMPLE_LEVELS:
        pct = int(lvl * 100)
        
        SCENARIOS_KNN.append({
            "name": f"B_Uniform_F_{pct}_oversample",
            "weights": "uniform",
            "oversample_noise": True,
            "noise_mode": "fixed",
            "noise_level": lvl
        })
        SCENARIOS_KNN.append({
            "name": f"B_Uniform_A_{pct}_oversample",
            "weights": "uniform",
            "oversample_noise": True,
            "noise_mode": "adaptive",
            "noise_level": lvl
        })
        
        SCENARIOS_KNN.append({
            "name": f"D_Distance_F_{pct}_oversample",
            "weights": "distance",
            "oversample_noise": True,
            "noise_mode": "fixed",
            "noise_level": lvl
        })
        SCENARIOS_KNN.append({
            "name": f"D_Distance_A_{pct}_oversample",
            "weights": "distance",
            "oversample_noise": True,
            "noise_mode": "adaptive",
            "noise_level": lvl
        })
    
    models_to_run = ["logistic", "rf", "knn", "mlp"]
    
    for model_name in models_to_run:
        if model_name == "knn":
            current_scenarios = SCENARIOS_KNN
        else:
            current_scenarios = SCENARIOS_GENERIC

        for sc in current_scenarios:
            train_baseline_scenario(
                model_name, X_matrix, y_vector, ordered_ids, patient_labels, output_dir, sc, golden_folds_dict=golden_folds_dict
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--agg_method", type=str, default="mean_std")
    parser.add_argument("--use_golden_splits", type=str, default=None, 
                        help="Path to common_splits_multiseed.json for scientifically valid comparison")
    
    args = parser.parse_args()
    run_baseline_experiment(args)