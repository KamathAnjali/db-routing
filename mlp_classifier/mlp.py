import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score
import joblib

# ── 1. Load & clean ──────────────────────────────────────────────────────────
df = pd.read_csv("labeled_dataset.csv")
df = df[df["label"] != "EXCLUDE"].copy()
df["label"] = df["label"].astype(int)

print(f"Usable rows : {len(df)}")
print(f"Class counts:\n{df['label'].value_counts()}\n")

# ── 2. Features & labels ─────────────────────────────────────────────────────
score_cols = [f"score_{i}" for i in range(10)]
X = df[score_cols ].values   #  10 scores 
y = df["label"].values

# ── 3. 80/20 train/test split (stratified) ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train : {len(X_train)}  |  Test : {len(X_test)}")
print(f"Train — 0:{(y_train==0).sum()}  1:{(y_train==1).sum()}")
print(f"Test  — 0:{(y_test==0).sum()}  1:{(y_test==1).sum()}\n")

# ── 4. Scale ─────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 5. Train MLP ─────────────────────────────────────────────────────────────
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

mlp.fit(X_train_s, y_train)
print(f"Stopped at iteration : {mlp.n_iter_}\n")

# # ── 6. Evaluate ───────────────────────────────────────────────────────────────
# preds = mlp.predict(X_test_s)
# proba = mlp.predict_proba(X_test_s)[:, 1]
proba = mlp.predict_proba(X_test_s)[:, 1]
preds = (proba >= 0.35).astype(int)

print(classification_report(y_test, preds, target_names=["Unambiguous(0)", "Ambiguous(1)"]))
print(f"Balanced accuracy : {balanced_accuracy_score(y_test, preds):.4f}")
print(f"ROC-AUC           : {roc_auc_score(y_test, proba):.4f}")

# ── 7. Save ───────────────────────────────────────────────────────────────────
joblib.dump(mlp,    "mlp_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nSaved mlp_model.pkl and scaler.pkl")