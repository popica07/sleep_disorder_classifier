import json
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "sleep_night_features_8.csv"

RANDOM_SEEDS = list(range(100))   
TEST_SIZE = 0.30

MIN_SAMPLES_PER_CLASS = 5

MAX_EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-3
HIDDEN_DIM = 128

PATIENCE = 15
MIN_IMPROVEMENT = 1e-4

USE_CLASS_WEIGHT = False

OUT_DIR = HERE / "MLP_100seeds_rareclass_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PERSEED_CSV = OUT_DIR / "mlp_per_seed.csv"
SUMMARY_JSON = OUT_DIR / "mlp_summary.json"

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
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


df = pd.read_csv(CSV_PATH)
print(f"Loaded CSV: {CSV_PATH}")
print("Initial shape:", df.shape)

if "label" not in df.columns:
    raise RuntimeError("Column 'label' not found in CSV!")

counts = df["label"].value_counts()
rare_labels = counts[counts < MIN_SAMPLES_PER_CLASS].index.tolist()

print("\nClass counts BEFORE filtering:")
print(counts.sort_index().to_string())

if rare_labels:
    print(f"\nRemoving classes with < {MIN_SAMPLES_PER_CLASS} samples:", sorted(rare_labels))
    df = df[~df["label"].isin(rare_labels)].reset_index(drop=True)
else:
    print(f"\nNo classes below {MIN_SAMPLES_PER_CLASS}. Nothing removed.")

counts_after = df["label"].value_counts()

print("\nClass counts AFTER filtering:")
print(counts_after.sort_index().to_string())
print("Shape after filtering:", df.shape)


X_df = df.drop(columns=["label", "label_str", "npz_file", "edf_filename"], errors="ignore")
X_df = X_df.apply(pd.to_numeric, errors="coerce")
X_df = X_df.replace([np.inf, -np.inf], np.nan)
X_df = X_df.fillna(X_df.mean(numeric_only=True))
X = X_df.values.astype(np.float32)

y_raw = df["label"].astype(int).values
unique_labels = np.sort(np.unique(y_raw))
label_to_index = {lab: i for i, lab in enumerate(unique_labels)}
y = np.array([label_to_index[v] for v in y_raw], dtype=int)

num_classes = len(unique_labels)

print("\nLabel mapping (original -> mapped):")
for lab in unique_labels:
    print(f"  {lab} -> {label_to_index[lab]}")
print("num_classes:", num_classes)
print("y min/max:", int(y.min()), int(y.max()))

if int(y.max()) != num_classes - 1:
    raise RuntimeError("Label mapping failed: y.max() must equal num_classes-1")

accs, f1s = [], []
per_seed_rows = []

print("\nRunning MLP over", len(RANDOM_SEEDS), "seeds")

for seed in RANDOM_SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=y
        )
        split_mode = "stratified"
    except ValueError as e:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=seed,
            shuffle=True
        )
        split_mode = f"non_stratified ({e})"

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    if int(y_train_t.max()) >= num_classes:
        raise RuntimeError(f"[Seed {seed}] label out of range")

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = MLP(input_dim=X_train.shape[1], hidden_dim=HIDDEN_DIM, num_classes=num_classes)

    if USE_CLASS_WEIGHT:
        cw = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(num_classes),
            y=y_train
        )
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    best_loss = float("inf")
    bad_epochs = 0
    epochs_used = 0

    for epoch in range(MAX_EPOCHS):
        epoch_losses = []

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        epochs_used = epoch + 1

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
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    accs.append(acc)
    f1s.append(macro_f1)

    per_seed_rows.append({
        "seed": int(seed),
        "split_mode": split_mode,
        "epochs_used": int(epochs_used),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
    })

acc_mean, acc_std = float(np.mean(accs)), float(np.std(accs))
f1_mean, f1_std = float(np.mean(f1s)), float(np.std(f1s))

print("\nMLP RESULTS (100 SEEDS, rare-class removal)")
print(f"Accuracy : {acc_mean:.4f} ± {acc_std:.4f}")
print(f"Macro-F1 : {f1_mean:.4f} ± {f1_std:.4f}")

pd.DataFrame(per_seed_rows).to_csv(PERSEED_CSV, index=False)
print("\nSaved per-seed CSV:", PERSEED_CSV)

summary = {
    "csv_path": str(CSV_PATH),
    "min_samples_per_class": int(MIN_SAMPLES_PER_CLASS),
    "removed_labels": [int(x) for x in rare_labels],
    "kept_original_labels": unique_labels.tolist(),
    "label_mapping_original_to_new": {int(k): int(v) for k, v in label_to_index.items()},
    "n_seeds": int(len(RANDOM_SEEDS)),
    "test_size": float(TEST_SIZE),
    "max_epochs": int(MAX_EPOCHS),
    "batch_size": int(BATCH_SIZE),
    "lr": float(LR),
    "hidden_dim": int(HIDDEN_DIM),
    "patience": int(PATIENCE),
    "use_class_weight": bool(USE_CLASS_WEIGHT),
    "accuracy_mean": acc_mean,
    "accuracy_std": acc_std,
    "macro_f1_mean": f1_mean,
    "macro_f1_std": f1_std,
}

with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Saved summary JSON:", SUMMARY_JSON)
print("Done.")
