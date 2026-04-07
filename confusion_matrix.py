# IS MEDIAN THE CORRECT THRESHOLD????

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_data(json_path: str) -> list[dict]:
    with open(json_path, "r") as f:
        return json.load(f)


def classify_record(record: dict) -> tuple[bool, float]:
    """
    Returns:
        is_correct: True if top-1 retrieved db matches correct_db
        entropy:    overall_entropy value
    """
    top1_db = record["top10"][0]["db_id"]
    correct_db = record["correct_db"]
    is_correct = top1_db == correct_db
    entropy = record["overall_entropy"]
    return is_correct, entropy


def build_confusion_data(records: list[dict]) -> dict:
    entropies = []
    correctness = []

    for rec in records:
        is_correct, entropy = classify_record(rec)
        entropies.append(entropy)
        correctness.append(is_correct)

    # Use median as threshold for high/low entropy
    median_entropy = 0.9
    print(f"Median entropy (threshold): {median_entropy:.4f}")

    # Build 2x2 counts
    # Rows: Entropy (High / Low), Cols: Classification (Correct / Wrong)
    counts = {
        ("Low Entropy",  "Correct"): 0,
        ("Low Entropy",  "Wrong"):   0,
        ("High Entropy", "Correct"): 0,
        ("High Entropy", "Wrong"):   0,
    }

    for entropy, is_correct in zip(entropies, correctness):
        entropy_label = "High Entropy" if entropy >= median_entropy else "Low Entropy"
        class_label   = "Correct" if is_correct else "Wrong"
        counts[(entropy_label, class_label)] += 1

    return counts, median_entropy, entropies, correctness


def plot_confusion_matrix(counts: dict, median_entropy: float, output_path: str = "confusion_matrix.png"):
    entropy_labels = ["Low Entropy", "High Entropy"]
    class_labels   = ["Correct", "Wrong"]

    matrix = np.array([
        [counts[("Low Entropy",  "Correct")], counts[("Low Entropy",  "Wrong")]],
        [counts[("High Entropy", "Correct")], counts[("High Entropy", "Wrong")]],
    ])

    total = matrix.sum()
    pct_matrix = matrix / total * 100  # percentage for annotation

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        matrix,
        annot=False,          # we'll add custom annotations below
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=entropy_labels,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        cbar_kws={"label": "Count"},
    )

    # Custom annotations: count + percentage
    for i in range(2):
        for j in range(2):
            count = matrix[i, j]
            pct   = pct_matrix[i, j]
            ax.text(
                j + 0.5, i + 0.5,
                f"{count}\n({pct:.1f}%)",
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="white" if count > matrix.max() * 0.6 else "black",
            )

    ax.set_xlabel("Classification", fontsize=12, labelpad=10)
    ax.set_ylabel("Entropy Level", fontsize=12, labelpad=10)
    ax.set_title(
        f"Confusion Matrix: Entropy vs Classification\n"
        f"(Entropy threshold = median = {median_entropy:.4f})",
        fontsize=13, pad=14,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to: {output_path}")


def print_summary(counts: dict, median_entropy: float, entropies: list, correctness: list):
    total      = len(entropies)
    n_correct  = sum(correctness)
    n_wrong    = total - n_correct
    n_high     = sum(1 for e in entropies if e >= median_entropy)
    n_low      = total - n_high

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total records       : {total}")
    print(f"Correct (top-1 hit) : {n_correct}  ({n_correct/total*100:.1f}%)")
    print(f"Wrong               : {n_wrong}  ({n_wrong/total*100:.1f}%)")
    print(f"Low entropy (<med)  : {n_low}  ({n_low/total*100:.1f}%)")
    print(f"High entropy (>=med): {n_high}  ({n_high/total*100:.1f}%)")
    print("\nConfusion matrix counts (rows=entropy, cols=classification):")
    print(f"  {'':20s} {'Correct':>10} {'Wrong':>10}")
    for elabel in ["Low Entropy", "High Entropy"]:
        c = counts[(elabel, "Correct")]
        w = counts[(elabel, "Wrong")]
        print(f"  {elabel:20s} {c:>10} {w:>10}")
    print("=" * 50)


def main():
    # ── Configure these two paths ──────────────────────────────────────────
    JSON_PATH   = "./llm_rerank_openrouter_reasoning.json"      # path to your input JSON file
    OUTPUT_IMG  = "confusion_matrix_500.png"   # where to save the plot
    # ───────────────────────────────────────────────────────────────────────

    if not Path(JSON_PATH).exists():
        raise FileNotFoundError(f"Input file not found: {JSON_PATH}")

    records = load_data(JSON_PATH)
    counts, median_entropy, entropies, correctness = build_confusion_data(records)
    print_summary(counts, median_entropy, entropies, correctness)
    plot_confusion_matrix(counts, median_entropy, output_path=OUTPUT_IMG)


if __name__ == "__main__":
    main()