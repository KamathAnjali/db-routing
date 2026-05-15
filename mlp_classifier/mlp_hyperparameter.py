import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score

# ── LOAD ────────────────────────────────────────────
df = pd.read_csv("labeled_dataset.csv")
df = df[df["label"] != "EXCLUDE"].copy()
df["label"] = df["label"].astype(int)

score_cols = [f"score_{i}" for i in range(10)]
X = df[score_cols].values
y = df["label"].values

# ── SPLIT: train (60), val (20), test (20) ───────────
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)
# 0.25 of 0.8 = 0.2 → so final = 60/20/20

# ── SCALE ───────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ── GRID SEARCH ─────────────────────────────────────
weights = [1.5, 2.0, 2.2, 2.5]
thresholds = [0.3, 0.35, 0.4, 0.45]

best_score = -1
best_config = None

print("\n=== HYPERPARAMETER SEARCH ===")

for weight in weights:
    # sample weights
    sample_weights = np.where(y_train == 1, weight, 1.0)

    # train model
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42,
    )

    mlp.fit(X_train_s, y_train, sample_weight=sample_weights)

    # probabilities on validation
    val_proba = mlp.predict_proba(X_val_s)[:, 1]

    for threshold in thresholds:
        val_preds = (val_proba >= threshold).astype(int)

        bal_acc = balanced_accuracy_score(y_val, val_preds)

        print(f"weight={weight}, threshold={threshold} → bal_acc={bal_acc:.4f}")

        if bal_acc > best_score:
            best_score = bal_acc
            best_config = (weight, threshold, mlp)

print("\n=== BEST CONFIG ===")
best_weight, best_threshold, best_model = best_config
print(f"Weight: {best_weight}")
print(f"Threshold: {best_threshold}")
print(f"Validation Balanced Acc: {best_score:.4f}")

# ── FINAL TEST EVALUATION ───────────────────────────
test_proba = best_model.predict_proba(X_test_s)[:, 1]
test_preds = (test_proba >= best_threshold).astype(int)

print("\n=== FINAL TEST PERFORMANCE ===")
print(classification_report(y_test, test_preds))

print("Balanced accuracy:", balanced_accuracy_score(y_test, test_preds))
print("ROC-AUC:", roc_auc_score(y_test, test_proba))
print("model_threshold:", best_threshold, "; weight:", best_weight)