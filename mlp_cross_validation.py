import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load
df = pd.read_csv("labeled_dataset.csv")
df = df[df["label"] != "EXCLUDE"].copy()
df["label"] = df["label"].astype(int)

score_cols = [f"score_{i}" for i in range(10)]
X = df[score_cols].values
y = df["label"].values

# Setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

recalls = []
precisions = []
f1s = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    print(f"\n=== Fold {fold+1} ===")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    sample_weights = np.where(y_train == 1, 2.2, 1.0)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=300,
        early_stopping=True,
        random_state=42
    )

    mlp.fit(X_train_s, y_train, sample_weight=sample_weights)

    proba = mlp.predict_proba(X_test_s)[:, 1]
    preds = (proba >= 0.4).astype(int)

    report = classification_report(y_test, preds, output_dict=True)

    recalls.append(report["1"]["recall"])
    precisions.append(report["1"]["precision"])
    f1s.append(report["1"]["f1-score"])

# Final stats
print("\n\nFINAL RESULTS (Ambiguous class):")
print(f"Recall   : {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"F1       : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")