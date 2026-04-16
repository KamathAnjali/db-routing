import subprocess
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score

# 🔧 thresholds to test
thresholds = [0.05, 0.1, 0.15, 0.2]

results = []

for th in thresholds:
    print("\n" + "="*50)
    print(f"Running for threshold = {th}")
    print("="*50)

    # ── STEP 1: Generate dataset ───────────────────────────────
    subprocess.run([
        "python3",
        "16th_april_part2.py",
        str(th)   # pass threshold as argument
    ])

    # ── STEP 2: Load dataset ───────────────────────────────────
    df = pd.read_csv("labeled_dataset.csv")
    df = df[df["label"] != "EXCLUDE"].copy()
    df["label"] = df["label"].astype(int)

    score_cols = [f"score_{i}" for i in range(10)]
    X = df[score_cols].values
    y = df["label"].values

    # ── STEP 3: Train/test split ───────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── STEP 4: Scale ──────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── STEP 5: Class weights ──────────────────────────────────
    sample_weights = np.where(y_train == 1, 2.2, 1.0)

    # ── STEP 6: Train MLP ──────────────────────────────────────
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

    # ── STEP 7: Predict (fixed threshold) ──────────────────────
    proba = mlp.predict_proba(X_test_s)[:, 1]
    preds = (proba >= 0.45).astype(int)

    report = classification_report(y_test, preds, output_dict=True)

    recall_1 = report["1"]["recall"]
    precision_1 = report["1"]["precision"]
    f1_1 = report["1"]["f1-score"]

    bal_acc = balanced_accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    results.append({
        "threshold": th,
        "recall_amb": recall_1,
        "precision_amb": precision_1,
        "f1_amb": f1_1,
        "balanced_acc": bal_acc,
        "roc_auc": auc
    })

# ── FINAL RESULTS ─────────────────────────────────────────────
results_df = pd.DataFrame(results)
print("\n\nFINAL COMPARISON:")
print(results_df.sort_values(by="recall_amb", ascending=False))