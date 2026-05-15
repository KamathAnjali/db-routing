import pandas as pd
import numpy as np
import json
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    balanced_accuracy_score
)

# ── LOAD FULL DATA (KEEP EVERYTHING) ─────────────────
df_full = pd.read_csv("labeled_dataset.csv")

# Ensure is_correct is boolean
df_full["is_correct"] = df_full["is_correct"].fillna(False).astype(bool)

# ── CREATE TRAINING DATA (REMOVE EXCLUDE) ────────────
df = df_full[df_full["label"] != "EXCLUDE"].copy()
df["label"] = df["label"].astype(int)

# ── FEATURES ────────────────────────────────────────
score_cols = [f"score_{i}" for i in range(10)]
X = df[score_cols].values
y = df["label"].values

# ── SPLIT (IMPORTANT: USE df INDEX BUT MAP BACK TO FULL) ──
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42, stratify=y
)

# ── SCALE ───────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── TRAIN ───────────────────────────────────────────
weight = 2.2
sample_weights = np.where(y_train == 1, weight, 1.0)

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

# ── PREDICT ─────────────────────────────────────────
threshold = 0.4
proba = mlp.predict_proba(X_test_s)[:, 1]
preds = (proba >= threshold).astype(int)

# ── DETECTOR ANALYSIS ───────────────────────────────
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

print("\n========== DETECTOR ANALYSIS ==========")
print(f"Total test samples (MLP subset): {len(y_test)}")

print("\n--- Breakdown ---")
print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")

precision = tp / (tp + fp)
recall = tp / (tp + fn)

print("\n--- Metrics ---")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

# ── CLASSIFICATION REPORT ───────────────────────────
print("\n=== CLASSIFICATION REPORT ===\n")
print(classification_report(y_test, preds, target_names=["Unambiguous(0)", "Ambiguous(1)"]))

print(f"Balanced accuracy: {balanced_accuracy_score(y_test, preds):.4f}")
print(f"ROC-AUC:           {roc_auc_score(y_test, proba):.4f}")

# ────────────────────────────────────────────────────
# 🔥 BASELINE ON TEST (INCLUDING ESTIMATED MISSES)
# ────────────────────────────────────────────────────

# usable test set
test_original = df.loc[test_idx]

usable_test = len(test_original)
correct_usable = test_original["is_correct"].sum()

# ---- estimate misses ----
total_full = 3000
miss_full = 197
usable_full = total_full - miss_full

miss_per_usable = miss_full / usable_full
estimated_miss = int(round(miss_per_usable * usable_test))

true_total = usable_test + estimated_miss

# ---- final baseline ----
true_baseline_acc = correct_usable / true_total

print("\n=== TRUE BASELINE (TEST SET) ===")
print(f"Usable test samples:     {usable_test}")
print(f"Estimated misses:        {estimated_miss}")
print(f"Total (with misses):     {true_total}")

print(f"\nCorrect (usable only):   {correct_usable}")
print(f"Baseline accuracy:       {true_baseline_acc:.4f}")
# ── AMBIGUOUS DETECTION OUTPUT ──────────────────────
test_df = df.loc[test_idx].copy()
test_df["ambiguous_prob"] = proba
test_df["is_ambiguous"] = preds

ambiguous_qids = set(test_df[test_df["is_ambiguous"] == 1]["question_id"])

print(f"\nAmbiguous detected: {len(ambiguous_qids)}")

# ── LOAD JSON ───────────────────────────────────────
with open("llm_rerank_openrouter_reasoning.json", "r") as f:
    full_data = json.load(f)

# ── FILTER ──────────────────────────────────────────
ambiguous_full = [
    item for item in full_data
    if item["question_id"] in ambiguous_qids
]

# ── SAVE ────────────────────────────────────────────
with open("ambiguous_full.json", "w") as f:
    json.dump(ambiguous_full, f, indent=2)

print(f"Saved {len(ambiguous_full)} enriched ambiguous queries")

fn_df = test_df[(preds == 0) & (y_test == 1)]
fn_correct = fn_df["is_correct"].sum()