import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "llm_rerank_openrouter_reasoning.json"

with open(INPUT_FILE) as f:
    data = json.load(f)

concentrations, top1_correct = [], []
for item in data:
    top10  = item["top10"]
    scores = [x["llm_score"] for x in top10]
    dbs    = [x["db_id"]     for x in top10]
    total  = sum(scores)
    conc   = scores[0] / total if total > 0 else 0.0
    concentrations.append(conc)
    top1_correct.append(dbs[0] == item["correct_db"])

concentrations = np.array(concentrations)
top1_correct   = np.array(top1_correct)

THRESHOLD = np.median(concentrations)
print(f"Concentration threshold (median) = {THRESHOLD:.4f}")

high_conc = concentrations >= THRESHOLD   # confident
n = len(concentrations)

# Row 0 = High Concentration (confident), Row 1 = Low Concentration (flagged)
# Col 0 = Correct, Col 1 = Wrong
matrix = np.array([
    [( high_conc & top1_correct).sum(),  ( high_conc & ~top1_correct).sum()],
    [(~high_conc & top1_correct).sum(),  (~high_conc & ~top1_correct).sum()],
])

tp = matrix[1, 1]  # low conc & wrong
fp = matrix[1, 0]  # low conc & correct
fn = matrix[0, 1]  # high conc & wrong
tn = matrix[0, 0]  # high conc & correct

precision   = tp / (tp + fp)
recall      = tp / (tp + fn)
f1          = 2 * precision * recall / (precision + recall)

print(f"Precision   : {precision:.3f}")
print(f"Recall      : {recall:.3f}")
print(f"F1          : {f1:.3f}")

fig, ax = plt.subplots(figsize=(7, 5.5))

sns.heatmap(
    matrix,
    annot=False,
    fmt="d",
    cmap="Blues",
    xticklabels=["Correct", "Wrong"],
    yticklabels=["High Concentration\n(Confident)", "Low Concentration\n(Flagged)"],
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
ax.set_ylabel("Concentration Level", fontsize=12)
ax.set_title(
    f"Confusion Matrix: Concentration vs Classification\n"
    f"(Concentration threshold = median = {THRESHOLD:.4f})",
    fontsize=12,
)

plt.tight_layout()
plt.savefig("confusion_matrix_concentration.png", dpi=150, bbox_inches="tight")
plt.show()