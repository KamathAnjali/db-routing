import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score

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

# Train
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=300,
    early_stopping=True,
    random_state=42,
)

mlp.fit(X_train_s, y_train)

proba = mlp.predict_proba(X_test_s)[:, 1]

# 🔥 SEARCH thresholds
best_t = 0.5
best_score = 0

for t in np.arange(0.2, 0.6, 0.05):
    preds = (proba >= t).astype(int)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    score = 0.7 * recall + 0.3 * f1  # prioritize recall

    if score > best_score:
        best_score = score
        best_t = t

print(f"Best threshold: {best_t}")