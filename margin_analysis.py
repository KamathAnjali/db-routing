"""
Ambiguity analysis for DB routing using top-2 score margin.
Goal: find a margin threshold to flag queries as ambiguous at inference time,
validated using gold labels (correct_db / correct_db_rank).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE = "llm_rerank_openrouter_reasoning.json"   # change path if needed
OUTPUT_DIR = Path(".")

# ── Load ──────────────────────────────────────────────────────────────────────
with open(INPUT_FILE) as f:
    data = json.load(f)

# ── Feature extraction ────────────────────────────────────────────────────────
rows = []
for item in data:
    top10 = item["top10"]
    scores = [x["llm_score"] for x in top10]
    dbs    = [x["db_id"]     for x in top10]

    s1 = scores[0] if len(scores) > 0 else 0.0
    s2 = scores[1] if len(scores) > 1 else 0.0
    margin = s1 - s2

    correct_db   = item["correct_db"]
    top1_correct = (dbs[0] == correct_db) if dbs else False
    correct_rank = item.get("correct_db_rank", None)   # 1-indexed

    rows.append({
        "question_id":   item["question_id"],
        "question":      item["question"],
        "correct_db":    correct_db,
        "top1_db":       dbs[0] if dbs else None,
        "top1_score":    s1,
        "top2_score":    s2,
        "margin":        margin,
        "entropy":       item.get("overall_entropy", np.nan),
        "correct_rank":  correct_rank,
        "top1_correct":  top1_correct,
    })

df = pd.DataFrame(rows)

# ── Basic stats ───────────────────────────────────────────────────────────────
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total queries        : {len(df)}")
print(f"Top-1 accuracy       : {df['top1_correct'].mean():.3f}  ({df['top1_correct'].sum()} / {len(df)})")
print(f"\nMargin (top1 - top2) statistics:")
print(df["margin"].describe().to_string())

# ── Threshold sweep ───────────────────────────────────────────────────────────
thresholds = np.round(np.arange(0.00, 0.31, 0.01), 3)

records = []
for t in thresholds:
    ambiguous   = df[df["margin"] < t]
    confident   = df[df["margin"] >= t]

    n_amb  = len(ambiguous)
    n_conf = len(confident)
    pct_amb = n_amb / len(df)

    acc_conf = confident["top1_correct"].mean() if n_conf > 0 else np.nan
    precision_flag = (~ambiguous["top1_correct"]).mean() if n_amb > 0 else np.nan
    wrong_total = (~df["top1_correct"]).sum()
    recall_flag = (~ambiguous["top1_correct"]).sum() / wrong_total if wrong_total > 0 else np.nan

    records.append({
        "threshold":       t,
        "n_ambiguous":     n_amb,
        "pct_flagged":     pct_amb,
        "acc_confident":   acc_conf,
        "precision_flag":  precision_flag,
        "recall_flag":     recall_flag,
        "f1_flag":         2 * precision_flag * recall_flag / (precision_flag + recall_flag)
                           if (precision_flag and recall_flag and (precision_flag + recall_flag) > 0)
                           else np.nan,
    })

sweep = pd.DataFrame(records)

print("\n" + "=" * 60)
print("THRESHOLD SWEEP  (margin < threshold → flag as ambiguous)")
print("=" * 60)
print(sweep.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ── Best threshold by F1 ──────────────────────────────────────────────────────
best_row = sweep.loc[sweep["f1_flag"].idxmax()]
T = best_row["threshold"]
print(f"\n→ Best threshold by F1 : {T:.2f}")
print(f"  Flags {best_row['pct_flagged']*100:.1f}% of queries as ambiguous")
print(f"  Precision (wrong|flagged) : {best_row['precision_flag']:.3f}")
print(f"  Recall    (flagged|wrong) : {best_row['recall_flag']:.3f}")
print(f"  Confident-set accuracy    : {best_row['acc_confident']:.3f}")

# ── Distribution split at best threshold ─────────────────────────────────────
df["flagged"] = df["margin"] < T

print(f"\n{'─'*60}")
print(f"SPLIT AT threshold = {T:.2f}")
print(f"{'─'*60}")
print(f"Confident  ({(~df['flagged']).sum():4d} queries): "
      f"top-1 acc = {df.loc[~df['flagged'], 'top1_correct'].mean():.3f}")
print(f"Ambiguous  ({df['flagged'].sum():4d} queries): "
      f"top-1 acc = {df.loc[df['flagged'], 'top1_correct'].mean():.3f}  ← these need disambiguator")

# ── Margin stats by correctness ───────────────────────────────────────────────
print(f"\nMargin stats — correct top-1:")
print(df.loc[df["top1_correct"],  "margin"].describe().to_string())
print(f"\nMargin stats — incorrect top-1:")
print(df.loc[~df["top1_correct"], "margin"].describe().to_string())

# ── Plots ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

# 1. Margin histogram by correctness
ax1 = fig.add_subplot(gs[0, :2])
bins = np.linspace(0, df["margin"].max() + 0.01, 40)
ax1.hist(df.loc[ df["top1_correct"],  "margin"], bins=bins, alpha=0.65,
         label="Top-1 correct",   color="#2ecc71", edgecolor="white")
ax1.hist(df.loc[~df["top1_correct"], "margin"], bins=bins, alpha=0.65,
         label="Top-1 incorrect", color="#e74c3c", edgecolor="white")
ax1.axvline(T, color="black", linestyle="--", linewidth=1.8, label=f"Best threshold ({T:.2f})")
ax1.set_xlabel("Margin (top1 score − top2 score)")
ax1.set_ylabel("Count")
ax1.set_title("Score Margin Distribution by Top-1 Correctness")
ax1.legend()

# 2. % flagged vs threshold
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(sweep["threshold"], sweep["pct_flagged"] * 100, color="#3498db", linewidth=2)
ax2.axvline(T, color="black", linestyle="--", linewidth=1.5)
ax2.set_xlabel("Margin threshold")
ax2.set_ylabel("% queries flagged")
ax2.set_title("Disambiguation Rate vs Threshold")

# 3. Precision / Recall / F1
ax3 = fig.add_subplot(gs[1, :2])
ax3.plot(sweep["threshold"], sweep["precision_flag"], label="Precision (wrong|flagged)", color="#e67e22", linewidth=2)
ax3.plot(sweep["threshold"], sweep["recall_flag"],    label="Recall (flagged|wrong)",    color="#9b59b6", linewidth=2)
ax3.plot(sweep["threshold"], sweep["f1_flag"],        label="F1",                        color="#2c3e50", linewidth=2.5, linestyle="--")
ax3.axvline(T, color="black", linestyle=":", linewidth=1.5, label=f"Best F1 @ {T:.2f}")
ax3.set_xlabel("Margin threshold")
ax3.set_ylabel("Score")
ax3.set_title("Flagging Quality vs Threshold")
ax3.legend(fontsize=8)
ax3.set_ylim(0, 1.05)

# 4. Confident-set accuracy vs threshold
ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(sweep["threshold"], sweep["acc_confident"], color="#1abc9c", linewidth=2)
ax4.axvline(T, color="black", linestyle="--", linewidth=1.5)
ax4.set_xlabel("Margin threshold")
ax4.set_ylabel("Top-1 accuracy")
ax4.set_title("Confident-Set Accuracy vs Threshold")
ax4.set_ylim(0, 1.05)

fig.suptitle("Ambiguity Flagging Analysis — Top-2 Score Margin", fontsize=14, fontweight="bold")
plot_path = OUTPUT_DIR / "ambiguity_analysis.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {plot_path}")
plt.show()

# ── Export flagged queries for inspection ─────────────────────────────────────
flagged_df = df[df["flagged"]].sort_values("margin")[
    ["question_id", "question", "correct_db", "top1_db", "top1_score", "top2_score", "margin", "top1_correct"]
]
csv_path = OUTPUT_DIR / "flagged_queries.csv"
flagged_df.to_csv(csv_path, index=False)
print(f"Flagged queries CSV → {csv_path}  ({len(flagged_df)} rows)")