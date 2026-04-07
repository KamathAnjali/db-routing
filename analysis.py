"""
analyse_checkpoint.py
---------------------
Full analysis of the LLM reranker checkpoint JSON.
Run: python analyse_checkpoint.py <path_to_json>
Produces: a PNG report + printed summary to terminal.
"""

import json
import sys
import re
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ── Load ──────────────────────────────────────────────────────────────────────
path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("llm_rerank_openrouter_2.json")
with open(path) as f:
    data = json.load(f)

# ── Feature extraction ────────────────────────────────────────────────────────
rows = []
for entry in data:
    scores = sorted([x["llm_score"] for x in entry["top5"]], reverse=True)
    rank   = entry.get("correct_db_rank")

    rows.append({
        "question_id":   entry["question_id"],
        "question":      entry["question"],
        "correct_db":    entry.get("correct_db"),
        "correct_rank":  rank,
        "correct":       rank == 1,
        "not_in_top5":   rank is None,
        "entropy":       entry["overall_entropy"],
        "top1_score":    scores[0],
        "top2_score":    scores[1] if len(scores) > 1 else 0.0,
        "margin":        scores[0] - (scores[1] if len(scores) > 1 else 0.0),
        "non_zero":      sum(1 for s in scores if s > 0),
    })

total        = len(rows)
correct      = sum(1 for r in rows if r["correct"])
wrong        = sum(1 for r in rows if not r["correct"] and not r["not_in_top5"])
not_in_top5  = sum(1 for r in rows if r["not_in_top5"])

# ── Terminal summary ──────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  CHECKPOINT ANALYSIS  —  {total} queries")
print(f"{'='*55}")
print(f"  Top-1 correct   : {correct}  ({correct/total:.1%})")
print(f"  Wrong (in top5) : {wrong}   ({wrong/total:.1%})")
print(f"  Not in top5     : {not_in_top5}  ({not_in_top5/total:.1%})")
print(f"  Avg entropy     : {np.mean([r['entropy'] for r in rows]):.4f}")
print(f"  Avg margin      : {np.mean([r['margin']  for r in rows]):.4f}")

# Rank distribution
rank_counts = Counter(r["correct_rank"] for r in rows)
print(f"\n  Rank distribution:")
for k in sorted(k for k in rank_counts if k is not None):
    print(f"    rank {k}: {rank_counts[k]}")
print(f"    not_found: {rank_counts[None]}")

# Most confused DB pairs
confused = defaultdict(int)
for r in rows:
    if not r["correct"] and not r["not_in_top5"]:
        # find what was ranked #1 instead
        entry   = next(e for e in data if e["question_id"] == r["question_id"])
        ranked1 = max(entry["top5"], key=lambda x: x["llm_score"])["db_id"]
        confused[(r["correct_db"], ranked1)] += 1

print(f"\n  Top confused pairs (correct_db → predicted_db):")
for (correct_db, predicted), cnt in sorted(confused.items(), key=lambda x: -x[1])[:10]:
    print(f"    {correct_db:25s} → {predicted:25s}  ({cnt}x)")

# ── Helpers ───────────────────────────────────────────────────────────────────
def bucket(values, labels, key):
    buckets = defaultdict(list)
    for r in rows:
        b = None
        for lo, hi, lab in labels:
            if lo <= r[key] < hi:
                b = lab
                break
        if b:
            buckets[b].append(r["correct"])
    return {b: (sum(v)/len(v) if v else 0, len(v)) for b, v in buckets.items()}

entropy_labels = [
    (0.0, 0.3, "0.0–0.3"),
    (0.3, 0.6, "0.3–0.6"),
    (0.6, 0.9, "0.6–0.9"),
    (0.9, 1.1, "0.9–1.1"),
    (1.1, 1.3, "1.1–1.3"),
    (1.3, 9.9, "1.3+"),
]
margin_labels = [
    (0.0,  0.1,  "0.0–0.1"),
    (0.1,  0.2,  "0.1–0.2"),
    (0.2,  0.3,  "0.2–0.3"),
    (0.3,  0.5,  "0.3–0.5"),
    (0.5,  0.7,  "0.5–0.7"),
    (0.7,  9.9,  "0.7+"),
]
top1_labels = [
    (0.0, 0.3,  "0.0–0.3"),
    (0.3, 0.5,  "0.3–0.5"),
    (0.5, 0.7,  "0.5–0.7"),
    (0.7, 0.9,  "0.7–0.9"),
    (0.9, 1.01, "0.9–1.0"),
]

entropy_acc = bucket(rows, entropy_labels, "entropy")
margin_acc  = bucket(rows, margin_labels,  "margin")
top1_acc    = bucket(rows, top1_labels,    "top1_score")

# ── Plot ──────────────────────────────────────────────────────────────────────
CORRECT_CLR = "#4ade80"
WRONG_CLR   = "#f87171"
NEUTRAL_CLR = "#60a5fa"
BG          = "#0f172a"
PANEL       = "#1e293b"
TEXT        = "#e2e8f0"
MUTED       = "#94a3b8"

fig = plt.figure(figsize=(18, 13), facecolor=BG)
fig.suptitle(
    f"LLM Reranker — Checkpoint Analysis  ({total} queries)",
    fontsize=15, color=TEXT, fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38,
                       left=0.06, right=0.97, top=0.93, bottom=0.06)

# ── 1. Accuracy summary bar ───────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
ax0.set_facecolor(PANEL)
labels_s = ["Correct\n(rank 1)", "Wrong\n(rank 2–5)", "Not in\ntop 5"]
vals_s   = [correct, wrong, not_in_top5]
colors_s = [CORRECT_CLR, WRONG_CLR, "#facc15"]
bars = ax0.bar(labels_s, vals_s, color=colors_s, width=0.5, edgecolor="none")
for bar, v in zip(bars, vals_s):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{v}\n({v/total:.0%})", ha="center", va="bottom",
             color=TEXT, fontsize=9, fontweight="bold")
ax0.set_title("Overall Accuracy", color=TEXT, fontsize=11, pad=8)
ax0.set_ylim(0, max(vals_s)*1.3)
ax0.tick_params(colors=MUTED)
for spine in ax0.spines.values(): spine.set_visible(False)
ax0.yaxis.set_visible(False)

# ── 2. Rank distribution ──────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
ax1.set_facecolor(PANEL)
rank_x = [str(k) if k is not None else "NF" for k in sorted(rank_counts, key=lambda x: (x is None, x))]
rank_y = [rank_counts[k] for k in sorted(rank_counts, key=lambda x: (x is None, x))]
rank_c = [CORRECT_CLR if k == "1" else (WRONG_CLR if k != "NF" else "#facc15") for k in rank_x]
ax1.bar(rank_x, rank_y, color=rank_c, edgecolor="none", width=0.5)
ax1.set_title("Rank Distribution of Correct DB", color=TEXT, fontsize=11, pad=8)
ax1.set_xlabel("Rank", color=MUTED, fontsize=9)
ax1.tick_params(colors=MUTED)
for spine in ax1.spines.values(): spine.set_visible(False)
ax1.yaxis.set_visible(False)
for i, v in enumerate(rank_y):
    ax1.text(i, v + 0.3, str(v), ha="center", va="bottom", color=TEXT, fontsize=9)

# ── 3. Entropy distribution: correct vs wrong ─────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor(PANEL)
ent_correct = [r["entropy"] for r in rows if r["correct"]]
ent_wrong   = [r["entropy"] for r in rows if not r["correct"]]
bins = np.linspace(0, 1.5, 16)
ax2.hist(ent_correct, bins=bins, alpha=0.7, color=CORRECT_CLR, label="Correct", edgecolor="none")
ax2.hist(ent_wrong,   bins=bins, alpha=0.7, color=WRONG_CLR,   label="Wrong",   edgecolor="none")
ax2.axvline(np.mean(ent_correct), color=CORRECT_CLR, linestyle="--", linewidth=1.2)
ax2.axvline(np.mean(ent_wrong),   color=WRONG_CLR,   linestyle="--", linewidth=1.2)
ax2.set_title("Entropy: Correct vs Wrong", color=TEXT, fontsize=11, pad=8)
ax2.set_xlabel("Entropy", color=MUTED, fontsize=9)
ax2.legend(facecolor=PANEL, edgecolor="none", labelcolor=TEXT, fontsize=8)
ax2.tick_params(colors=MUTED)
for spine in ax2.spines.values(): spine.set_visible(False)
ax2.yaxis.set_visible(False)

# ── 4. Accuracy by entropy bin ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(PANEL)
e_bins = list(entropy_acc.keys())
e_acc  = [entropy_acc[b][0] for b in e_bins]
e_cnt  = [entropy_acc[b][1] for b in e_bins]
bar_colors = [CORRECT_CLR if a >= 0.5 else WRONG_CLR for a in e_acc]
ax3.bar(range(len(e_bins)), e_acc, color=bar_colors, edgecolor="none", width=0.6)
ax3.set_xticks(range(len(e_bins)))
ax3.set_xticklabels(e_bins, rotation=30, ha="right", fontsize=8, color=MUTED)
for i, (a, c) in enumerate(zip(e_acc, e_cnt)):
    ax3.text(i, a + 0.02, f"{a:.0%}\nn={c}", ha="center", va="bottom",
             color=TEXT, fontsize=8)
ax3.axhline(0.5, color=MUTED, linestyle="--", linewidth=0.8, alpha=0.5)
ax3.set_ylim(0, 1.2)
ax3.set_title("Top-1 Accuracy by Entropy Bin", color=TEXT, fontsize=11, pad=8)
ax3.tick_params(colors=MUTED)
for spine in ax3.spines.values(): spine.set_visible(False)
ax3.yaxis.set_visible(False)

# ── 5. Accuracy by margin bin ─────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(PANEL)
m_bins = list(margin_acc.keys())
m_acc  = [margin_acc[b][0] for b in m_bins]
m_cnt  = [margin_acc[b][1] for b in m_bins]
bar_colors = [CORRECT_CLR if a >= 0.5 else WRONG_CLR for a in m_acc]
ax4.bar(range(len(m_bins)), m_acc, color=bar_colors, edgecolor="none", width=0.6)
ax4.set_xticks(range(len(m_bins)))
ax4.set_xticklabels(m_bins, rotation=30, ha="right", fontsize=8, color=MUTED)
for i, (a, c) in enumerate(zip(m_acc, m_cnt)):
    ax4.text(i, a + 0.02, f"{a:.0%}\nn={c}", ha="center", va="bottom",
             color=TEXT, fontsize=8)
ax4.axhline(0.5, color=MUTED, linestyle="--", linewidth=0.8, alpha=0.5)
ax4.set_ylim(0, 1.2)
ax4.set_title("Top-1 Accuracy by Margin Bin", color=TEXT, fontsize=11, pad=8)
ax4.tick_params(colors=MUTED)
for spine in ax4.spines.values(): spine.set_visible(False)
ax4.yaxis.set_visible(False)

# ── 6. Accuracy by top1 score bin ─────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor(PANEL)
t_bins = list(top1_acc.keys())
t_acc  = [top1_acc[b][0] for b in t_bins]
t_cnt  = [top1_acc[b][1] for b in t_bins]
bar_colors = [CORRECT_CLR if a >= 0.5 else WRONG_CLR for a in t_acc]
ax5.bar(range(len(t_bins)), t_acc, color=bar_colors, edgecolor="none", width=0.6)
ax5.set_xticks(range(len(t_bins)))
ax5.set_xticklabels(t_bins, rotation=30, ha="right", fontsize=8, color=MUTED)
for i, (a, c) in enumerate(zip(t_acc, t_cnt)):
    ax5.text(i, a + 0.02, f"{a:.0%}\nn={c}", ha="center", va="bottom",
             color=TEXT, fontsize=8)
ax5.axhline(0.5, color=MUTED, linestyle="--", linewidth=0.8, alpha=0.5)
ax5.set_ylim(0, 1.2)
ax5.set_title("Top-1 Accuracy by Winner Score", color=TEXT, fontsize=11, pad=8)
ax5.tick_params(colors=MUTED)
for spine in ax5.spines.values(): spine.set_visible(False)
ax5.yaxis.set_visible(False)

# ── 7. Scatter: margin vs entropy, coloured by correctness ────────────────────
ax6 = fig.add_subplot(gs[2, 0:2])
ax6.set_facecolor(PANEL)
for r in rows:
    color  = CORRECT_CLR if r["correct"] else (WRONG_CLR if not r["not_in_top5"] else "#facc15")
    marker = "o" if not r["not_in_top5"] else "x"
    ax6.scatter(r["entropy"], r["margin"], color=color, alpha=0.55,
                s=30, marker=marker, edgecolors="none")

# draw threshold guide lines
ax6.axvline(0.9,  color=MUTED, linestyle="--", linewidth=0.8, alpha=0.5, label="entropy=0.9")
ax6.axhline(0.25, color=NEUTRAL_CLR, linestyle="--", linewidth=0.8, alpha=0.5, label="margin=0.25")

# legend proxies
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=CORRECT_CLR, markersize=7, label='Correct'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=WRONG_CLR,   markersize=7, label='Wrong'),
    Line2D([0], [0], marker='x', color='#facc15', markersize=7,            label='Not in top5'),
]
ax6.legend(handles=legend_elements, facecolor=PANEL, edgecolor="none",
           labelcolor=TEXT, fontsize=8, loc="upper right")
ax6.set_xlabel("Entropy", color=MUTED, fontsize=9)
ax6.set_ylabel("Margin (top1 − top2)", color=MUTED, fontsize=9)
ax6.set_title("Entropy vs Margin — coloured by correctness", color=TEXT, fontsize=11, pad=8)
ax6.tick_params(colors=MUTED)
for spine in ax6.spines.values(): spine.set_color("#334155")

# Annotate quadrants
ax6.text(0.05, 0.92, "Low entropy\nHigh margin\n→ Safe to pass through",
         transform=ax6.transAxes, color=CORRECT_CLR, fontsize=7.5, alpha=0.8,
         verticalalignment="top")
ax6.text(0.60, 0.12, "High entropy\nLow margin\n→ Flag for Stage 2",
         transform=ax6.transAxes, color=WRONG_CLR, fontsize=7.5, alpha=0.8,
         verticalalignment="bottom")

# ── 8. Top confused DB pairs ──────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
ax7.set_facecolor(PANEL)
ax7.axis("off")
ax7.set_title("Top Confused DB Pairs\n(correct → predicted)", color=TEXT, fontsize=11, pad=8)

top_confused = sorted(confused.items(), key=lambda x: -x[1])[:8]
if top_confused:
    y = 0.93
    for (correct_db, predicted), cnt in top_confused:
        ax7.text(0.02, y, f"{correct_db}", color=CORRECT_CLR, fontsize=8,
                 transform=ax7.transAxes, va="top")
        ax7.text(0.42, y, "→", color=MUTED, fontsize=8,
                 transform=ax7.transAxes, va="top")
        ax7.text(0.48, y, f"{predicted}", color=WRONG_CLR, fontsize=8,
                 transform=ax7.transAxes, va="top")
        ax7.text(0.88, y, f"×{cnt}", color=TEXT, fontsize=8, fontweight="bold",
                 transform=ax7.transAxes, va="top")
        y -= 0.11
else:
    ax7.text(0.5, 0.5, "No wrong predictions", color=MUTED,
             ha="center", va="center", transform=ax7.transAxes)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = "./outputs/checkpoint_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\n  Plot saved → {out_path}")

# ── Print threshold recommendations ──────────────────────────────────────────
print(f"\n{'='*55}")
print("  THRESHOLD RECOMMENDATIONS")
print(f"{'='*55}")

# Find entropy threshold where accuracy drops below 50%
for lo, hi, lab in entropy_labels:
    acc, cnt = entropy_acc.get(lab, (1.0, 0))
    flag = " ← FLAG" if acc < 0.5 else ""
    print(f"  entropy {lab:8s}  acc={acc:.0%}  n={cnt}{flag}")

print()
for lo, hi, lab in margin_labels:
    acc, cnt = margin_acc.get(lab, (1.0, 0))
    flag = " ← FLAG" if acc < 0.5 else ""
    print(f"  margin  {lab:8s}  acc={acc:.0%}  n={cnt}{flag}")

# Suggest combined rule
print(f"\n  SUGGESTED FLAG RULE:")
print(f"    flag if entropy > 0.9  OR  margin < 0.25")
flagged      = [r for r in rows if r["entropy"] > 0.9 or r["margin"] < 0.25]
flagged_wrong = [r for r in flagged if not r["correct"]]
not_flagged_wrong = [r for r in rows if not r["correct"] and not (r["entropy"] > 0.9 or r["margin"] < 0.25)]
print(f"    Queries flagged      : {len(flagged)}/{total} ({len(flagged)/total:.0%})")
print(f"    Wrong caught by flag : {len(flagged_wrong)}/{wrong+not_in_top5} ({len(flagged_wrong)/(wrong+not_in_top5) if (wrong+not_in_top5) else 0:.0%})")
print(f"    Wrong missed by flag : {len(not_flagged_wrong)}")
print(f"{'='*55}\n")