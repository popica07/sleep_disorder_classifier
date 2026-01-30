import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

CSV_PATH = "/Maastricht/Sleep disorder/sleep_night_features_8.csv"

RANDOM_SEED = 42
TEST_SIZE = 0.30
N_ESTIMATORS = 400

OUT_DIR = "/Maastricht/Sleep disorder/RF"
os.makedirs(OUT_DIR, exist_ok=True)

RESULTS_JSON = os.path.join(
    OUT_DIR, "rf_results_train_test_sleep_night_features_8.json"
)

CM_PNG = os.path.join(
    OUT_DIR, "confusion_matrix_rf_train_test_sleep_night_features_8.png"
)

ACC_PNG = os.path.join(
    OUT_DIR, "rf_accuracy_train_test_sleep_night_features_8.png"
)

F1_PNG = os.path.join(
    OUT_DIR, "rf_macrof1_train_test_sleep_night_features_8.png"
)

df = pd.read_csv(CSV_PATH)
print("Loaded CSV:", CSV_PATH)
print("Shape:", df.shape)

if "label" not in df.columns:
    raise RuntimeError("Column 'label' not found in CSV!")

y = df["label"].astype(int)
X = df.drop(
    columns=["label", "label_str", "npz_file", "edf_filename"],
    errors="ignore"
)

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean(numeric_only=True))

if "label_str" in df.columns:
    label_map = (
        df[["label", "label_str"]]
        .drop_duplicates()
        .sort_values("label")
        .set_index("label")
    )
    labels_sorted = sorted(label_map.index.astype(int).tolist())
    class_names = [label_map.loc[l, "label_str"] for l in labels_sorted]
else:
    labels_sorted = sorted(np.unique(y).tolist())
    class_names = [str(l) for l in labels_sorted]

print("\nClass distribution:")
if "label_str" in df.columns:
    print(df["label_str"].value_counts())
else:
    print(pd.Series(y).value_counts().sort_index())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

print(f"\nTrain size: {len(X_train)}")
print(f"Test size : {len(X_test)}")

clf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

print("\n=== RANDOM FOREST (TRAIN/TEST SPLIT) ===")
print("Accuracy :", round(acc, 4))
print("Macro F1 :", round(macro_f1, 4))
print("\nConfusion Matrix:\n", cm)

results = {
    "csv_path": CSV_PATH,
    "random_seed": RANDOM_SEED,
    "test_size": TEST_SIZE,
    "n_estimators": N_ESTIMATORS,
    "accuracy": float(acc),
    "macro_f1": float(macro_f1),
    "labels_sorted": [int(x) for x in labels_sorted],
    "class_names": class_names,
    "confusion_matrix": cm.tolist(),
}

with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved results JSON to: {RESULTS_JSON}")

plt.figure()
plt.bar(["Random Forest"], [acc])
plt.ylabel("Accuracy")
plt.title("Random Forest Accuracy (Train/Test Split)")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(ACC_PNG, dpi=300)
plt.close()

print(f"Saved accuracy plot to: {ACC_PNG}")

plt.figure()
plt.bar(["Random Forest"], [macro_f1])
plt.ylabel("Macro-F1")
plt.title("Random Forest Macro-F1 (Train/Test Split)")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(F1_PNG, dpi=300)
plt.close()

print(f"Saved Macro-F1 plot to: {F1_PNG}")

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
plt.title("Confusion Matrix – Random Forest (Train/Test)")
plt.tight_layout()
plt.savefig(CM_PNG, dpi=300)
plt.close()

print(f"Saved confusion matrix image to: {CM_PNG}")
