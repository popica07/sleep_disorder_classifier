import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

RANDOM_SEED = 42
TEST_SIZE = 0.30

K = 3
WEIGHTS = "distance"

CSV_PATH = "/Maastricht/Sleep disorder/sleep_night_features_8.csv"
CM_PNG = "/Maastricht/Sleep disorder/confusion_matrix_knn_optionB.png"

ACC_PNG = "/Maastricht/Sleep disorder/KNN_again_results/knn_accuracy_vs_k.png"
F1_PNG = "/Maastricht/Sleep disorder/KNN_again_results/knn_macrof1_vs_k.png"

K_RANGE = list(range(3, 22, 2))


def main():
    df = pd.read_csv(CSV_PATH)
    print("Loaded:", CSV_PATH)
    print("Shape :", df.shape)

    if "label" not in df.columns:
        raise RuntimeError("Missing column: label")

    if "label_str" in df.columns:
        print("\nClass distribution (label_str):")
        print(df["label_str"].value_counts())

    y = df["label"].astype(int)

    X = df.drop(columns=["label", "label_str", "npz_file", "edf_filename"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean(numeric_only=True))

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=y
        )
        split_mode = "stratified"
    except ValueError as e:
        print("\n Stratified split failed, using NON-stratified split.")
        print("Reason:", e)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True
        )
        split_mode = "non_stratified"

    print(f"\nSplit mode: {split_mode}")
    print("Train size:", len(X_train))
    print("Test size :", len(X_test))

    ks_used = []
    accs = []
    macro_f1s = []

    for k in K_RANGE:
        if k > len(X_train):
            continue

        model_k = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, weights=WEIGHTS))
        ])
        model_k.fit(X_train, y_train)
        y_pred_k = model_k.predict(X_test)

        acc_k = accuracy_score(y_test, y_pred_k)
        f1_k = f1_score(y_test, y_pred_k, average="macro", zero_division=0)

        ks_used.append(k)
        accs.append(acc_k)
        macro_f1s.append(f1_k)

    plt.figure()
    plt.plot(ks_used, accs, marker="o")
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Accuracy")
    plt.title(f"KNN Accuracy vs k (Train/Test split, {split_mode}, weights={WEIGHTS})")
    plt.xticks(ks_used)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ACC_PNG, dpi=300)
    plt.close()
    print("Saved accuracy vs k plot to:", ACC_PNG)

    plt.figure()
    plt.plot(ks_used, macro_f1s, marker="o")
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Macro-F1")
    plt.title(f"KNN Macro-F1 vs k (Train/Test split, {split_mode}, weights={WEIGHTS})")
    plt.xticks(ks_used)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(F1_PNG, dpi=300)
    plt.close()
    print("Saved macro-F1 vs k plot to:", F1_PNG)

    if len(macro_f1s) > 0:
        best_idx = int(np.argmax(macro_f1s))
        print(f"\nBest k on this split by Macro-F1: k={ks_used[best_idx]} | Macro-F1={macro_f1s[best_idx]:.4f} | Acc={accs[best_idx]:.4f}")

    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=K, weights=WEIGHTS))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    labels_sorted = np.sort(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    print("\n=== KNN (TRAIN/TEST SPLIT) ===")
    print(f"K        : {K} (weights={WEIGHTS})")
    print("Accuracy :", round(acc, 3))
    print("Macro F1 :", round(macro_f1, 3))
    print("\nConfusion Matrix:\n", cm)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, labels=labels_sorted, zero_division=0))

    if "label_str" in df.columns:
        label_map = (
            df[["label", "label_str"]]
            .drop_duplicates()
            .set_index("label")
        )
        class_names = [label_map.loc[int(l), "label_str"] for l in labels_sorted]
    else:
        class_names = [str(l) for l in labels_sorted]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix – KNN (Train/Test, k={K})")
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=300)
    plt.close()

    print("\nSaved confusion matrix to:", CM_PNG)


if __name__ == "__main__":
    main()
