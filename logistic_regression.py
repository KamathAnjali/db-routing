import json
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── STEP 1: Load your JSON ──────────────────────────────
with open("llm_rerank_openrouter_reasoning.json", "r") as f:
    data = json.load(f)

# ── STEP 2: Calculate features for each query ──────────
rows = []

for item in data:
    scores = [db["llm_score"] for db in item["top10"]]
    top1   = scores[0]
    top2   = scores[1]

    margin        = top1 - top2
    concentration = top1 / (sum(scores) + 1e-9)  # how dominant is top1
    entropy       = item["overall_entropy"]       # already computed for you!

    # Label: 1 = wrong retrieval, 0 = correct
    label = 0 if item["correct_db_rank"] == 1 else 1

    rows.append({
        "entropy":       entropy,
        "margin":        margin,
        "top1_score":    top1,
        "concentration": concentration,
        "label":         label
    })

df = pd.DataFrame(rows)
print(df["label"].value_counts())  # check how many correct vs wrong you have

# ── STEP 3: Train the classifier ───────────────────────
X = df[["entropy", "margin", "top1_score", "concentration"]].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)

# ── STEP 4: Evaluate ────────────────────────────────────
probs       = model.predict_proba(X_test)[:, 1]
predictions = (probs > 0.35).astype(int)  # lower threshold = higher recall

print(classification_report(y_test, predictions,
      target_names=["Correct", "Wrong"]))