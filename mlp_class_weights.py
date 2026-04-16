import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score

# Load
df = pd.read_csv("labeled_dataset.csv")
df = df[df["label"] != "EXCLUDE"].copy()
df["label"] = df["label"].astype(int)

score_cols = [f"score_{i}" for i in range(10)]
X = df[score_cols].values
y = df["label"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 🔥 CHANGE: sample weights
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

# Default threshold

model_threshold = 0.4
proba = mlp.predict_proba(X_test_s)[:, 1]
preds = (proba >= model_threshold).astype(int)

print(classification_report(y_test, preds))
print("Balanced accuracy:", balanced_accuracy_score(y_test, preds))
print("ROC-AUC:", roc_auc_score(y_test, proba))
print("model_threshold:",model_threshold, "; weight:", weight)