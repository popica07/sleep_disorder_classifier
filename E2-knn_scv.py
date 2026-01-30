import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

RANDOM_SEED = 42
N_FOLDS = 5
MIN_SAMPLES_PER_CLASS = 5

K_VALUES = [3, 5, 7, 9, 11, 15, 21]
WEIGHTS = "distance"

BASE_DIR = "/Maastricht/Sleep disorder"
CSV_PATH = os.path.join(BASE_DIR, "sleep_night_features_8.csv")

OUT_DIR = os.path.join(BASE_DIR, "KNN")
os.makedirs(OUT_DIR, exist_ok=True)

RESULTS_JSON = os.path.join(OUT_DIR, "knn_5fold_cv_drop_rare_sleep_night_features_8.json")
CM_PNG = os.path.join(OUT_DIR, "confusion_matrix_knn_sleep_night_features_8.png")
DROPPED_CSV = os.path.join(OUT_DIR, "dropped_samples_rare_classes_knn_sleep_night_features_8.csv")
KEPT_CSV = os.path.join(OUT_DIR, "kept_samples_after_drop_knn_sleep_night_features_8.csv")

ACC_PNG = os.path.join(OUT_DIR, "knn_5fold_accuracy_vs_k.png")
F1_PNG = os.path.join(OUT_DIR, "knn_5fold_macrof1_vs_k.png")


if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print("Loaded CSV:", CSV_PATH)
print("Initial shape:", df.shape)

META_COLS = [c for c in ["npz_file", "edf_filename", "label_str", "label"] if c in df.columns]
if "label" not in df.columns:
    raise RuntimeError("Column 'label' not found in CSV!")

X = df.drop(columns=["label", "label_str", "npz_file", "edf_filename"], errors="ignore")
y = df["label"].astype(int)

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
nan_before = int(X.isna().sum().sum())
X = X.fillna(X.mean(numeric_only=True))
nan_after = int(X.isna().sum().sum())

print("\nAfter cleaning:")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("NaNs before:", nan_before, "NaNs after:", nan_after)

if "label_str" in df.columns:
    print("\nClass counts BEFORE dropping:")
    print(df["label_str"].value_counts())

class_counts = y.value_counts().sort_index()
rare_labels = class_counts[class_counts < MIN_SAMPLES_PER_CLASS].index.tolist()

if rare_labels:
    print(f"\nDropping rare classes (<{MIN_SAMPLES_PER_CLASS} samples): {rare_labels}")
    print("Counts dropped:")
    print(class_counts.loc[rare_labels])

    keep_mask = ~y.isin(rare_labels)

    dropped_df = df.loc[~keep_mask, META_COLS].copy() if META_COLS else df.loc[~keep_mask, :].copy()
    kept_df = df.loc[keep_mask, META_COLS].copy() if META_COLS else df.loc[keep_mask, :].copy()

    dropped_df.to_csv(DROPPED_CSV, index=False)
    kept_df.to_csv(KEPT_CSV, index=False)

    print(f"Saved dropped log: {DROPPED_CSV}")
    print(f"Saved kept log   : {KEPT_CSV}")

    df = df.loc[keep_mask].reset_index(drop=True)
    X = X.loc[keep_mask].reset_index(drop=True)
    y = y.loc[keep_mask].reset_index(drop=True)
else:
    print(f"\nNo classes below {MIN_SAMPLES_PER_CLASS}. Nothing dropped.")
    if META_COLS:
        df[META_COLS].to_csv(KEPT_CSV, index=False)
        print(f"Saved kept log   : {KEPT_CSV}")

print("\nAfter dropping:")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Class counts (numeric):")
print(y.value_counts().sort_index())

min_class = int(y.value_counts().min())
if min_class < N_FOLDS:
    raise ValueError(
        f"Cannot do {N_FOLDS}-fold stratified CV: min class count is {min_class}. "
        f"Increase MIN_SAMPLES_PER_CLASS or reduce N_FOLDS."
    )

all_labels = np.sort(y.unique())

if "label_str" in df.columns:
    label_map = (
        df[["label", "label_str"]]
        .drop_duplicates()
        .set_index("label")["label_str"]
        .to_dict()
    )
    class_names = [label_map[int(l)] for l in all_labels]
else:
    class_names = [str(int(l)) for l in all_labels]


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

fold_details = []
test_accs = []
test_f1s = []
k_chosen = []

cm_total = np.zeros((len(all_labels), len(all_labels)), dtype=int)

acc_by_k = {k: [] for k in K_VALUES}
f1_by_k = {k: [] for k in K_VALUES}

print("\nKNN 5-FOLD STRATIFIED CV (best k per fold by macro-F1)")

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    valid_k = [k for k in K_VALUES if k <= len(X_train)]
    if not valid_k:
        raise ValueError(f"Fold {fold}: train size is {len(X_train)} so no valid k in {K_VALUES}")

    best_k = None
    best_f1 = -1.0
    best_acc = -1.0
    best_pred = None

    for k in valid_k:
        model = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, weights=WEIGHTS))
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", labels=all_labels, zero_division=0)

        acc_by_k[k].append(acc)
        f1_by_k[k].append(macro_f1)

        if (macro_f1 > best_f1) or (macro_f1 == best_f1 and acc > best_acc):
            best_f1 = macro_f1
            best_acc = acc
            best_k = k
            best_pred = y_pred

    cm_fold = confusion_matrix(y_test, best_pred, labels=all_labels)
    cm_total += cm_fold

    test_accs.append(best_acc)
    test_f1s.append(best_f1)
    k_chosen.append(best_k)

    fold_details.append({
        "fold": fold,
        "best_k": int(best_k),
        "test_accuracy": float(best_acc),
        "test_macro_f1": float(best_f1),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "valid_k_values": [int(k) for k in valid_k],
    })

    print(f"\n--- Fold {fold}/{N_FOLDS} ---")
    print(f"Valid k: {valid_k}")
    print(f"Best k={best_k} | acc={best_acc:.4f} | macroF1={best_f1:.4f}")


ks_used = [k for k in K_VALUES if len(acc_by_k[k]) > 0]

acc_mean = [float(np.mean(acc_by_k[k])) for k in ks_used]
acc_std = [float(np.std(acc_by_k[k])) for k in ks_used]

f1_mean = [float(np.mean(f1_by_k[k])) for k in ks_used]
f1_std = [float(np.std(f1_by_k[k])) for k in ks_used]

plt.figure()
plt.plot(ks_used, acc_mean, marker="o", label="Mean accuracy")
plt.fill_between(
    ks_used,
    np.array(acc_mean) - np.array(acc_std),
    np.array(acc_mean) + np.array(acc_std),
    alpha=0.2,
    label="±1 std"
)
plt.xticks(ks_used)
plt.xlabel("k (number of neighbors)")
plt.ylabel("Accuracy")
plt.title(f"KNN Accuracy vs k ({N_FOLDS}-Fold Stratified CV, weights={WEIGHTS})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(ACC_PNG, dpi=300)
plt.close()
print(f"\nSaved accuracy-vs-k plot: {ACC_PNG}")

plt.figure()
plt.plot(ks_used, f1_mean, marker="o", label="Mean macro-F1")
plt.fill_between(
    ks_used,
    np.array(f1_mean) - np.array(f1_std),
    np.array(f1_mean) + np.array(f1_std),
    alpha=0.2,
    label="±1 std"
)
plt.xticks(ks_used)
plt.xlabel("k (number of neighbors)")
plt.ylabel("Macro-F1")
plt.title(f"KNN Macro-F1 vs k ({N_FOLDS}-Fold Stratified CV, weights={WEIGHTS})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(F1_PNG, dpi=300)
plt.close()
print(f"Saved macroF1-vs-k plot: {F1_PNG}")

results = {
    "csv_path": CSV_PATH,
    "random_seed": int(RANDOM_SEED),
    "model": "KNN",
    "weights": WEIGHTS,
    "k_values_tested": K_VALUES,
    "n_folds": int(N_FOLDS),
    "min_samples_per_class_threshold": int(MIN_SAMPLES_PER_CLASS),
    "dropped_labels": [int(x) for x in rare_labels],
    "labels_used": [int(x) for x in all_labels],
    "class_names": class_names,
    "chosen_k_per_fold": [int(k) for k in k_chosen],
    "test_accuracy_mean": float(np.mean(test_accs)),
    "test_accuracy_std": float(np.std(test_accs)),
    "test_macro_f1_mean": float(np.mean(test_f1s)),
    "test_macro_f1_std": float(np.std(test_f1s)),
    "confusion_matrix_sum_over_tests": cm_total.tolist(),
    "fold_details": fold_details,
    "dropped_samples_csv": DROPPED_CSV if rare_labels else None,
    "kept_samples_csv": KEPT_CSV,
    "per_k_accuracy_mean": {int(k): float(np.mean(acc_by_k[k])) for k in ks_used},
    "per_k_accuracy_std": {int(k): float(np.std(acc_by_k[k])) for k in ks_used},
    "per_k_macro_f1_mean": {int(k): float(np.mean(f1_by_k[k])) for k in ks_used},
    "per_k_macro_f1_std": {int(k): float(np.std(f1_by_k[k])) for k in ks_used},
}

print("\n FINAL (mean over folds, using best k per fold)")
print(f"Accuracy mean: {results['test_accuracy_mean']:.4f} ± {results['test_accuracy_std']:.4f}")
print(f"Macro-F1 mean: {results['test_macro_f1_mean']:.4f} ± {results['test_macro_f1_std']:.4f}")

with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved results JSON: {RESULTS_JSON}")

white_green = LinearSegmentedColormap.from_list("white_green", ["#ffffff", "#2ca02c"])

plt.figure(figsize=(10, 8))
vmax = max(10, int(cm_total.max()))
plt.imshow(cm_total, cmap=white_green, interpolation="nearest", vmin=0, vmax=vmax)
plt.colorbar()

plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
plt.yticks(np.arange(len(class_names)), class_names)

max_val = cm_total.max() if cm_total.max() > 0 else 1
for i in range(cm_total.shape[0]):
    for j in range(cm_total.shape[1]):
        val = int(cm_total[i, j])
        plt.text(
            j, i, val,
            ha="center", va="center",
            color="black" if val < max_val * 0.6 else "white"
        )

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title(f"Confusion Matrix – KNN (5-Fold CV, weights={WEIGHTS}) – sleep_night_features_8.csv")
plt.tight_layout()
plt.savefig(CM_PNG, dpi=300)
plt.close()

print(f"Saved confusion matrix PNG: {CM_PNG}")
