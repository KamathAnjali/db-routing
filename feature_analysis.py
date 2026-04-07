import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def load(path):
    with open(path) as f:
        return json.load(f)


def get_bucket(record, median_entropy):
    top1 = record["top10"][0]["db_id"]
    correct = record["correct_db"]
    is_correct = top1 == correct
    entropy = record["overall_entropy"]
    e_label = "HE" if entropy >= median_entropy else "LE"
    c_label = "correct" if is_correct else "wrong"
    return f"{e_label}-{c_label}"


def extract_features(record):
    scores = [x["llm_score"] for x in record["top10"]]
    top1_score = scores[0]
    top2_score = scores[1] if len(scores) > 1 else 0.0

    # --- Tier 1 ---
    margin = top1_score - top2_score

    live = sum(1 for s in scores if s > 0.05)

    total = sum(scores)
    concentration = top1_score / total if total > 0 else 0.0

    # --- Tier 2 ---
    correct_db_rank = record.get("correct_db_rank")  # 1-indexed or null

    survived_step3 = 10 - len(record["reasoning"].get("step2_eliminated", []))

    gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
    biggest_gap_at = int(np.argmax(gaps)) if gaps else 0  # 0-indexed

    return {
        "question_id":       record["question_id"],
        "entropy":           record["overall_entropy"],
        "margin":            round(margin, 4),
        "live_candidates":   live,
        "concentration":     round(concentration, 4),
        "correct_db_rank":   correct_db_rank,
        "survived_step3":    survived_step3,
        "biggest_gap_at":    biggest_gap_at,
    }


def bucket_stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n":    len(values),
        "mean": round(float(np.mean(values)), 4),
        "std":  round(float(np.std(values)), 4),
        "min":  round(float(np.min(values)), 4),
        "max":  round(float(np.max(values)), 4),
    }


def print_table(feature_name, bucket_data):
    buckets = ["LE-correct", "LE-wrong", "HE-correct", "HE-wrong"]
    print(f"\n{'─'*70}")
    print(f"  {feature_name}")
    print(f"{'─'*70}")
    print(f"  {'Bucket':<16} {'n':>4}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
    print(f"  {'─'*16} {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for b in buckets:
        s = bucket_data.get(b, {})
        if s.get("n", 0) == 0:
            print(f"  {b:<16} {'0':>4}  {'—':>8}")
        else:
            print(f"  {b:<16} {s['n']:>4}  {s['mean']:>8}  {s['std']:>8}  {s['min']:>8}  {s['max']:>8}")


def main():
    JSON_PATH = "./llm_rerank_openrouter_reasoning.json"

    if not Path(JSON_PATH).exists():
        raise FileNotFoundError(f"Not found: {JSON_PATH}")

    records = load(JSON_PATH)

    entropies = [r["overall_entropy"] for r in records]
    median_entropy = float(np.median(entropies))
    print(f"\nMedian entropy (bucket threshold): {median_entropy:.4f}")
    print(f"Total records: {len(records)}")

    features_by_bucket = defaultdict(list)
    all_features = []

    for rec in records:
        bucket = get_bucket(rec, median_entropy)
        feats = extract_features(rec)
        feats["bucket"] = bucket
        all_features.append(feats)
        features_by_bucket[bucket].append(feats)

    feature_names = [
        "entropy",
        "margin",
        "live_candidates",
        "concentration",
        "correct_db_rank",
        "survived_step3",
        "biggest_gap_at",
    ]

    print("\n" + "=" * 70)
    print("  FEATURE STATISTICS BY CONFUSION MATRIX BUCKET")
    print("=" * 70)

    for fname in feature_names:
        bucket_data = {}
        for bucket, recs in features_by_bucket.items():
            vals = [r[fname] for r in recs]
            bucket_data[bucket] = bucket_stats(vals)
        print_table(fname, bucket_data)

    print(f"\n{'─'*70}")
    print("  BUCKET SIZES")
    print(f"{'─'*70}")
    for b in ["LE-correct", "LE-wrong", "HE-correct", "HE-wrong"]:
        print(f"  {b:<16}  {len(features_by_bucket[b])} records")

    print(f"\n{'─'*70}")
    print("  CORRECT_DB_RANK DISTRIBUTION (wrong buckets only)")
    print(f"{'─'*70}")
    for b in ["LE-wrong", "HE-wrong"]:
        ranks = [r["correct_db_rank"] for r in features_by_bucket[b]]
        null_count = sum(1 for r in ranks if r is None)
        ranked = sorted([r for r in ranks if r is not None])
        print(f"  {b}: null={null_count}, ranks={ranked}")

    print(f"\n{'─'*70}")
    print("  BIGGEST_GAP_AT DISTRIBUTION (where does confidence drop off?)")
    print(f"{'─'*70}")
    for b in ["LE-correct", "LE-wrong", "HE-correct", "HE-wrong"]:
        gaps = [r["biggest_gap_at"] for r in features_by_bucket[b]]
        from collections import Counter
        dist = sorted(Counter(gaps).items())
        print(f"  {b:<16}: {dict(dist)}")

    print()


if __name__ == "__main__":
    main()