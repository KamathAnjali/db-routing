import pandas as pd
import numpy as np
import json
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── LOAD CSV ────────────────────────────────────────
df = pd.read_csv("labeled_dataset.csv")
df = df[df["label"] != "EXCLUDE"].copy()
df["label"] = df["label"].astype(int)

# ── FEATURES ────────────────────────────────────────
score_cols = [f"score_{i}" for i in range(10)]
X = df[score_cols].values
y = df["label"].values

# ── SPLIT (keep indices!) ───────────────────────────
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

# ── MAP BACK TO ORIGINAL DF ─────────────────────────
test_df = df.loc[test_idx].copy()
test_df["ambiguous_prob"] = proba
test_df["is_ambiguous"] = preds

# ── GET AMBIGUOUS QIDs ──────────────────────────────
ambiguous_qids = set(test_df[test_df["is_ambiguous"] == 1]["question_id"])

print(f"Ambiguous detected: {len(ambiguous_qids)}")

# ── LOAD FULL JSON ──────────────────────────────────
with open("llm_rerank_openrouter_reasoning.json", "r") as f:
    full_data = json.load(f)

# ── FILTER MATCHING ENTRIES ─────────────────────────
ambiguous_full = [
    item for item in full_data
    if item["question_id"] in ambiguous_qids
]

# ── SAVE ────────────────────────────────────────────
with open("ambiguous_full.json", "w") as f:
    json.dump(ambiguous_full, f, indent=2)

print(f"Saved {len(ambiguous_full)} enriched ambiguous queries")

from sklearn.metrics import confusion_matrix

# ── CONFUSION MATRIX ────────────────────────────────
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

print("\n========== DETECTOR ANALYSIS ==========")

print(f"Total test samples: {len(y_test)}")

print("\n--- Ground Truth ---")
print(f"Actual ambiguous:     {tp + fn}")
print(f"Actual unambiguous:   {tn + fp}")

print("\n--- Model Output ---")
print(f"Predicted ambiguous:  {tp + fp}")   # this is your 199
print(f"Predicted clear:      {tn + fn}")

print("\n--- Breakdown ---")
print(f"TP (correct ambiguous): {tp}")
print(f"FP (false ambiguous):   {fp}")
print(f"FN (missed ambiguous):  {fn}")
print(f"TN (correct clear):     {tn}")

# ── METRICS ─────────────────────────────────────────
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print("\n--- Metrics ---")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score

# ── PREDICT ─────────────────────────────────────────
threshold = 0.4
proba = mlp.predict_proba(X_test_s)[:, 1]
preds = (proba >= threshold).astype(int)

# ── BASELINE BEFORE CLASSIFIER (TEST SET ONLY) ───────
test_original = df.loc[test_idx]

total_test = len(test_original)
baseline_correct = test_original["is_correct"].sum()
baseline_accuracy = baseline_correct / total_test

print("\n=== BASELINE (ORIGINAL SYSTEM ON TEST SET) ===")
print(f"Total test samples:     {total_test}")
print(f"Correct (baseline):     {baseline_correct}")
print(f"Incorrect (baseline):   {total_test - baseline_correct}")
print(f"Baseline accuracy:      {baseline_accuracy:.4f}")

# ── PRINT METRICS ───────────────────────────────────
print("\n=== CLASSIFICATION REPORT ===\n")
print(classification_report(y_test, preds, target_names=["Unambiguous(0)", "Ambiguous(1)"]))

print(f"Balanced accuracy: {balanced_accuracy_score(y_test, preds):.4f}")
print(f"ROC-AUC:           {roc_auc_score(y_test, proba):.4f}")
print(f"model_threshold: {threshold} ; weight: {weight}")