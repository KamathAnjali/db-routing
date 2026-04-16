"""
label_dataset.py
----------------
Labeling rules:
    UNAMBIGUOUS (0): correctly classified AND clear margin (margin >= threshold)
    AMBIGUOUS   (1): (misclassified AND gold in top 10) AND (correctly classified and margin < threshold)
    EXCLUDE:         gold not in top-10 (retrieval miss)
                     
"""

import json
import csv

# ── CONFIG ────────────────────────────────────────────────────────────────────

INPUT_FILE       = "llm_rerank_openrouter_reasoning.json"
OUTPUT_FILE      = "labeled_dataset.csv"
MARGIN_THRESHOLD = 0.3

# ── LABELING ──────────────────────────────────────────────────────────────────

def assign_label(entry, threshold):
    gold_db    = entry["correct_db"]
    top10      = entry["top10"]                        # list of {db_id, llm_score}

    # Sort descending by llm_score (defensive)
    top10      = sorted(top10, key=lambda x: x["llm_score"], reverse=True)
    scores     = [item["llm_score"] for item in top10]
    db_ids     = [item["db_id"]     for item in top10]

    is_correct = (db_ids[0] == gold_db)
    margin     = scores[0] - scores[1] if len(scores) > 1 else 1.0
    close      = margin < threshold

    gold_rank  = next(
        (i + 1 for i, db in enumerate(db_ids) if db == gold_db),
        None
    )

    if is_correct:
        if close:
            label, reason = 1, "correct_close"
        else:
            label, reason = 0, "correct_clear"
    else:
        if gold_rank is None:
            label, reason = "EXCLUDE", "retrieval_miss"
        elif close:
            label, reason = 1, "wrong_close"
        else:
            label, reason = "EXCLUDE", "wrong_clear_llm_error"

    return {
        "label":      label,
        "reason":     reason,
        "is_correct": is_correct,
        "gold_rank":  gold_rank,
        "margin":     round(margin, 6),
        "scores":     scores,
    }

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} queries.")

    rows = []
    for entry in data:
        r = assign_label(entry, MARGIN_THRESHOLD)
        row = {
            "question_id": entry["question_id"],
            "label":       r["label"],
            "reason":      r["reason"],
            "is_correct":  r["is_correct"],
            "gold_rank":   r["gold_rank"],
            "margin":      r["margin"],
        }
        for i, s in enumerate(r["scores"]):
            row[f"score_{i}"] = round(s, 6)
        rows.append(row)

    # ── Summary ───────────────────────────────────────────────────────────────
    from collections import Counter
    by_reason = Counter(r["reason"] for r in rows)

    excluded  = [r for r in rows if r["label"] == "EXCLUDE"]
    ambiguous = [r for r in rows if r["label"] == 1]
    unamb     = [r for r in rows if r["label"] == 0]
    usable    = len(ambiguous) + len(unamb)

    print(f"\nTotal queries:        {len(rows)}")
    print(f"\nBreakdown by reason:")
    for reason, count in by_reason.most_common():
        print(f"  {reason:35s}  {count}")

    print(f"\nExcluded:             {len(excluded)}")
    print(f"Usable for MLP:       {usable}")
    if usable > 0:
        print(f"  Ambiguous  (1):     {len(ambiguous)}  ({100*len(ambiguous)/usable:.1f}%)")
        print(f"  Unambiguous(0):     {len(unamb)}  ({100*len(unamb)/usable:.1f}%)")
        ratio = max(len(ambiguous), len(unamb)) / (min(len(ambiguous), len(unamb)) + 1e-9)
        print(f"\nClass ratio:          {ratio:.2f}:1")
        if ratio > 3.0:
            print("  WARNING: severe imbalance — use class_weight in training")
        elif ratio > 1.5:
            print("  Mild imbalance — monitor per-class F1")
        else:
            print("  Balanced")

    fieldnames = list(rows[0].keys())
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved → {OUTPUT_FILE}")
    print("Filter out EXCLUDE rows in training: df = df[df['label'] != 'EXCLUDE']")

if __name__ == "__main__":
    main()