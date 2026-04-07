import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "llm_rerank_openrouter_reasoning.json"
THRESHOLD  = 0.21

with open(INPUT_FILE) as f:
    data = json.load(f)

margins, top1_correct = [], []
for item in data:
    top10 = item["top10"]
    scores = [x["llm_score"] for x in top10]
    dbs    = [x["db_id"]     for x in top10]
    s1 = scores[0] if scores else 0.0
    s2 = scores[1] if len(scores) > 1 else 0.0
    margins.append(s1 - s2)
    top1_correct.append(dbs[0] == item["correct_db"])

margins      = np.array(margins)
top1_correct = np.array(top1_correct)
flagged      = margins < THRESHOLD
n            = len(margins)

# Row 0 = High Margin (confident), Row 1 = Low Margin (flagged)
# Col 0 = Correct,                 Col 1 = Wrong
matrix = np.array([
    [(~flagged & top1_correct).sum(),  (~flagged & ~top1_correct).sum()],   # High Margin
    [( flagged & top1_correct).sum(),  ( flagged & ~top1_correct).sum()],   # Low Margin
])

fig, ax = plt.subplots(figsize=(7, 5.5))

sns.heatmap(
    matrix,
    annot=False,
    fmt="d",
    cmap="Blues",
    xticklabels=["Correct", "Wrong"],
    yticklabels=["High Margin\n(Confident)", "Low Margin\n(Flagged)"],
    linewidths=0.5,
    linecolor="gray",
    ax=ax,
    cbar_kws={"label": "Count"},
)

for i in range(2):
    for j in range(2):
        count = matrix[i, j]
        text_color = "white" if count > matrix.max() * 0.55 else "black"
        ax.text(
            j + 0.5, i + 0.5,
            f"{count}\n({count/n*100:.1f}%)",
            ha="center", va="center",
            fontsize=13, fontweight="bold",
            color=text_color,
        )

ax.set_xlabel("Classification", fontsize=12)
ax.set_ylabel("Margin Level", fontsize=12)
ax.set_title(f"Confusion Matrix: Margin vs Classification\n(Margin threshold = {THRESHOLD})", fontsize=12)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()