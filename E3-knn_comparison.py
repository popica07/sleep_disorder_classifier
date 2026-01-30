import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

RANDOM_SEED = 42
TEST_SIZE = 0.30

N_FOLDS = 5
MIN_SAMPLES_PER_CLASS = 5

K_VALUES = [3, 5, 7, 9, 11, 15, 21]
WEIGHTS = "distance"

BASE_DIR = "/Maastricht/Sleep disorder"
CSV_PATH = os.path.join(BASE_DIR, "sleep_night_features_8.csv")

OUT_DIR = os.path.join(BASE_DIR, "KNN_compare_split_vs_cv")
os.makedirs(OUT_DIR, exist_ok=True)

PLOT_ACC = os.path.join(OUT_DIR, "compare_accuracy_split_vs_cv.png")
PLOT_F1 = os.path.join(OUT_DIR, "compare_macrof1_split_vs_cv.png")

CM_SPLIT_PNG = os.path.join(OUT_DIR, "cm_knn_train_test.png")
CM_CV_PNG = os.path.join(OUT_DIR, "cm_knn_5fold_cv_summed.png")


def load_and_clean(csv_path: str):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise RuntimeError("Missing column: label")

    y = df["label"].astype(int)

    X = df.drop(columns=["label", "label_str", "npz_file", "edf_filename"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean(numeric_only=True))

    return df, X, y


def drop_rare_classes(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, min_samples: int):
    class_counts = y.value_counts().sort_index()
    rare_labels = class_counts[class_counts < min_samples].index.tolist()

    if not rare_labels:
        return df, X, y, rare_labels

    keep_mask = ~y.isin(rare_labels)
    df2 = df.loc[keep_mask].reset_index(drop=True)
    X2 = X.loc[keep_mask].reset_index(drop=True)
    y2 = y.loc[keep_mask].reset_index(drop=True)

    return df2, X2, y2, rare_labels


def make_model(k: int):
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights=WEIGHTS))
    ])


def plot_confusion_matrix(cm: np.ndarray, labels, title: str, out_path: str):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved confusion matrix:", out_path)

def main():
    print("Loading:", CSV_PATH)
    df, X, y = load_and_clean(CSV_PATH)
    print("Initial shape:", df.shape)

    if "label_str" in df.columns:
        print("\nClass distribution (before drop):")
        print(df["label_str"].value_counts())

    df, X, y, dropped = drop_rare_classes(df, X, y, MIN_SAMPLES_PER_CLASS)
    print("\nAfter dropping rare classes (<{}):".format(MIN_SAMPLES_PER_CLASS))
    print("Shape:", X.shape)
    if dropped:
        print("Dropped labels:", dropped)

    min_class = int(y.value_counts().min())
    if min_class < N_FOLDS:
        raise ValueError(
            f"Cannot do {N_FOLDS}-fold stratified CV: min class count is {min_class}. "
            f"Increase MIN_SAMPLES_PER_CLASS or reduce N_FOLDS."
        )

    labels_sorted = np.sort(y.unique())

    if "label_str" in df.columns:
        label_map = (
            df[["label", "label_str"]]
            .drop_duplicates()
            .set_index("label")["label_str"]
            .to_dict()
        )
        class_names = [label_map[int(l)] for l in labels_sorted]
    else:
        class_names = [str(int(l)) for l in labels_sorted]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    acc_split = []
    f1_split = []

    for k in K_VALUES:
        if k > len(X_train):
            acc_split.append(np.nan)
            f1_split.append(np.nan)
            continue

        model = make_model(k)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        acc_split.append(accuracy_score(y_test, pred))
        f1_split.append(f1_score(y_test, pred, average="macro", zero_division=0))

    best_k_split = K_VALUES[int(np.nanargmax(f1_split))]
    print(f"\nBest k on Train/Test split (by Macro-F1): {best_k_split}")

    model_split = make_model(best_k_split)
    model_split.fit(X_train, y_train)
    pred_split = model_split.predict(X_test)
    cm_split = confusion_matrix(y_test, pred_split, labels=labels_sorted)

    plot_confusion_matrix(
        cm_split,
        class_names,
        title=f"KNN Confusion Matrix – Train/Test (k={best_k_split}, weights={WEIGHTS})",
        out_path=CM_SPLIT_PNG
    )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    acc_cv = {k: [] for k in K_VALUES}
    f1_cv = {k: [] for k in K_VALUES}

    cm_cv_total = np.zeros((len(labels_sorted), len(labels_sorted)), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        for k in K_VALUES:
            if k > len(X_tr):
                continue
            model = make_model(k)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)

            acc_cv[k].append(accuracy_score(y_te, pred))
            f1_cv[k].append(f1_score(y_te, pred, average="macro", zero_division=0))

        best_k_fold = None
        best_f1_fold = -1.0
        best_acc_fold = -1.0
        best_pred_fold = None

        valid_k = [k for k in K_VALUES if k <= len(X_tr)]
        for k in valid_k:
            model = make_model(k)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)

            acc = accuracy_score(y_te, pred)
            mf1 = f1_score(y_te, pred, average="macro", zero_division=0)

            if (mf1 > best_f1_fold) or (mf1 == best_f1_fold and acc > best_acc_fold):
                best_f1_fold = mf1
                best_acc_fold = acc
                best_k_fold = k
                best_pred_fold = pred

        cm_fold = confusion_matrix(y_te, best_pred_fold, labels=labels_sorted)
        cm_cv_total += cm_fold

        print(f"Fold {fold}/{N_FOLDS}: best_k={best_k_fold} | acc={best_acc_fold:.4f} | macroF1={best_f1_fold:.4f}")

    acc_cv_mean = [float(np.mean(acc_cv[k])) if len(acc_cv[k]) else np.nan for k in K_VALUES]
    acc_cv_std  = [float(np.std(acc_cv[k]))  if len(acc_cv[k]) else np.nan for k in K_VALUES]
    f1_cv_mean  = [float(np.mean(f1_cv[k]))  if len(f1_cv[k]) else np.nan for k in K_VALUES]
    f1_cv_std   = [float(np.std(f1_cv[k]))   if len(f1_cv[k]) else np.nan for k in K_VALUES]

    plot_confusion_matrix(
        cm_cv_total,
        class_names,
        title=f"KNN Confusion Matrix – {N_FOLDS}-Fold CV (summed, best k per fold, weights={WEIGHTS})",
        out_path=CM_CV_PNG
    )

    plt.figure()
    plt.plot(K_VALUES, acc_split, marker="o", label="Train/Test (1 split)")
    plt.plot(K_VALUES, acc_cv_mean, marker="s", linestyle="--", label=f"{N_FOLDS}-Fold CV (mean)")
    plt.fill_between(
        K_VALUES,
        np.array(acc_cv_mean) - np.array(acc_cv_std),
        np.array(acc_cv_mean) + np.array(acc_cv_std),
        alpha=0.2,
        label="CV ±1 std"
    )
    plt.xticks(K_VALUES)
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Accuracy")
    plt.title("KNN Accuracy: 1 Split vs 5-Fold CV")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_ACC, dpi=300)
    plt.close()

    plt.figure()
    plt.plot(K_VALUES, f1_split, marker="o", label="Train/Test (1 split)")
    plt.plot(K_VALUES, f1_cv_mean, marker="s", linestyle="--", label=f"{N_FOLDS}-Fold CV (mean)")
    plt.fill_between(
        K_VALUES,
        np.array(f1_cv_mean) - np.array(f1_cv_std),
        np.array(f1_cv_mean) + np.array(f1_cv_std),
        alpha=0.2,
        label="CV ±1 std"
    )
    plt.xticks(K_VALUES)
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Macro-F1")
    plt.title("KNN Macro-F1: 1 Split vs 5-Fold CV")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_F1, dpi=300)
    plt.close()

    print("\nSaved plots:")
    print(" -", PLOT_ACC)
    print(" -", PLOT_F1)
    print("\nSaved confusion matrices:")
    print(" -", CM_SPLIT_PNG)
    print(" -", CM_CV_PNG)

    best_k_cv = K_VALUES[int(np.nanargmax(f1_cv_mean))]
    print("\nBest k by Macro-F1:")
    print(f" - Train/Test split: k={best_k_split}, Macro-F1={np.nanmax(f1_split):.4f}")
    print(f" - 5-Fold CV mean  : k={best_k_cv}, Macro-F1={np.nanmax(f1_cv_mean):.4f}")


if __name__ == "__main__":
    main()
