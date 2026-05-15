import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# ── Load ─────────────────────────────────────────────
df = pd.read_csv("labeled_dataset.csv")
df = df[df["label"] != "EXCLUDE"].copy()
df["label"] = df["label"].astype(int)

score_cols = [f"score_{i}" for i in range(10)]
X = df[score_cols].values
y = df["label"].values

# ── Split ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scale ────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 🔥 Fixed best settings
sample_weights = np.where(y_train == 1, 2.2, 1.0)
threshold = 0.4

# 🔥 Architectures to test
architectures = [
    (32,),
    (64,),
    (64,32),
    (128,64),
    (128,64,32),
    (64,64),
    (128,)
]

results = []

for arch in architectures:
    print(f"\n=== Testing architecture: {arch} ===")

    mlp = MLPClassifier(
        hidden_layer_sizes=arch,
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

    proba = mlp.predict_proba(X_test_s)[:, 1]
    preds = (proba >= threshold).astype(int)

    report = classification_report(y_test, preds, output_dict=True)

    recall_1 = report["1"]["recall"]
    precision_1 = report["1"]["precision"]
    f1_1 = report["1"]["f1-score"]

    score = 0.7 * recall_1 + 0.3 * f1_1

    results.append({
        "architecture": str(arch),
        "recall": recall_1,
        "precision": precision_1,
        "f1": f1_1,
        "score": score
    })

# ── Results ──────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="score", ascending=False)

print("\n\nTOP ARCHITECTURES:")
print(results_df)