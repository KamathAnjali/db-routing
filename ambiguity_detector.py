import json
import numpy as np
from pathlib import Path
from itertools import product as iproduct


def load(path):
    with open(path) as f:
        return json.load(f)


def extract_features(record, median_entropy):
    scores = [x["llm_score"] for x in record["top10"]]
    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else 0.0
    total = sum(scores)

    margin        = top1 - top2
    concentration = top1 / total if total > 0 else 0.0
    entropy       = record["overall_entropy"]
    live          = sum(1 for s in scores if s > 0.05)
    survived      = 10 - len(record["reasoning"].get("step2_eliminated", []))

    gaps          = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
    gap_at        = int(np.argmax(gaps)) if gaps else 0

    is_correct    = record["top10"][0]["db_id"] == record["correct_db"]

    return {
        "question_id":   record["question_id"],
        "question":      record["question"],
        "is_correct":    is_correct,
        "entropy":       entropy,
        "margin":        margin,
        "concentration": concentration,
        "live":          live,
        "survived":      survived,
        "gap_at":        gap_at,
    }


def evaluate_rule(features, rule_fn):
    tp = fp = tn = fn = 0
    flagged_ids = []
    missed_ids  = []

    for f in features:
        flagged   = rule_fn(f)
        is_wrong  = not f["is_correct"]

        if flagged and is_wrong:
            tp += 1
        elif flagged and not is_wrong:
            fp += 1
            flagged_ids.append(f["question_id"])
        elif not flagged and is_wrong:
            fn += 1
            missed_ids.append((f["question_id"], f["question"][:60]))
        else:
            tn += 1

    total_wrong  = tp + fn
    total_flagged = tp + fp

    precision = tp / total_flagged if total_flagged > 0 else 0.0
    recall    = tp / total_wrong   if total_wrong   > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "flagged":   total_flagged,
        "missed_ids": missed_ids,
        "false_positive_ids": flagged_ids,
    }


def print_result(name, r, show_details=False):
    print(f"\n  {'─'*60}")
    print(f"  {name}")
    print(f"  {'─'*60}")
    print(f"  Flagged for disambig : {r['flagged']:>3}  (TP={r['tp']} FP={r['fp']})")
    print(f"  Silent failures      : {r['fn']:>3}  (wrong queries NOT flagged)")
    print(f"  Precision            : {r['precision']:.1%}")
    print(f"  Recall               : {r['recall']:.1%}")
    print(f"  F1                   : {r['f1']:.3f}")
    if show_details and r["missed_ids"]:
        print(f"\n  Silent failures (id | question):")
        for qid, q in r["missed_ids"]:
            print(f"    [{qid}] {q}")


def grid_search(features, margin_thresholds, concentration_thresholds, gap_at_thresholds):
    """Exhaustive search over threshold combinations."""
    best_f1    = -1
    best_prec  = -1
    best_combo = None
    results    = []

    for m_thresh, c_thresh, g_thresh in iproduct(
        margin_thresholds, concentration_thresholds, gap_at_thresholds
    ):
        def rule(f, m=m_thresh, c=c_thresh, g=g_thresh):
            return f["margin"] < m or f["concentration"] < c or f["gap_at"] >= g

        r = evaluate_rule(features, rule)
        results.append((m_thresh, c_thresh, g_thresh, r))

        if r["f1"] > best_f1 or (r["f1"] == best_f1 and r["precision"] > best_prec):
            best_f1    = r["f1"]
            best_prec  = r["precision"]
            best_combo = (m_thresh, c_thresh, g_thresh, r)

    return best_combo, results


def main():
    JSON_PATH = "./llm_rerank_openrouter_reasoning.json"

    if not Path(JSON_PATH).exists():
        raise FileNotFoundError(f"Not found: {JSON_PATH}")

    records  = load(JSON_PATH)
    entropies = [r["overall_entropy"] for r in records]
    median_e  = float(np.median(entropies))

    features = [extract_features(r, median_e) for r in records]

    total       = len(features)
    total_wrong = sum(1 for f in features if not f["is_correct"])
    print(f"\n{'='*62}")
    print(f"  AMBIGUITY DETECTOR — RULE EVALUATION")
    print(f"{'='*62}")
    print(f"  Total queries : {total}")
    print(f"  Wrong routing : {total_wrong}  ({total_wrong/total:.1%})")
    print(f"  Correct       : {total - total_wrong}  ({(total-total_wrong)/total:.1%})")

    # ── Baseline: entropy only ────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  BASELINES")

    print_result(
        "Baseline A — entropy > median",
        evaluate_rule(features, lambda f: f["entropy"] >= median_e)
    )

    print_result(
        "Baseline B — margin < 0.5",
        evaluate_rule(features, lambda f: f["margin"] < 0.5)
    )

    # ── Proposed combination rules ────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  COMBINATION RULES (derived from feature analysis)")

    print_result(
        "Rule 1 — margin < 0.5  OR  gap_at > 0",
        evaluate_rule(features,
            lambda f: f["margin"] < 0.5 or f["gap_at"] > 0),
        show_details=True
    )

    print_result(
        "Rule 2 — margin < 0.5  OR  (gap_at > 0 AND margin < 0.7)",
        evaluate_rule(features,
            lambda f: f["margin"] < 0.5 or (f["gap_at"] > 0 and f["margin"] < 0.7)),
        show_details=True
    )

    print_result(
        "Rule 3 — margin < 0.5  OR  concentration < 0.6",
        evaluate_rule(features,
            lambda f: f["margin"] < 0.5 or f["concentration"] < 0.6),
        show_details=True
    )

    print_result(
        "Rule 4 — margin < 0.7  AND  concentration < 0.7",
        evaluate_rule(features,
            lambda f: f["margin"] < 0.7 and f["concentration"] < 0.7),
        show_details=True
    )

    print_result(
        "Rule 5 — margin < 0.5  OR  gap_at > 0  OR  concentration < 0.55",
        evaluate_rule(features,
            lambda f: f["margin"] < 0.5 or f["gap_at"] > 0 or f["concentration"] < 0.55),
        show_details=True
    )

    # ── Grid search ───────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  GRID SEARCH — best threshold combination")
    print("  Rule form: margin < M  OR  concentration < C  OR  gap_at >= G")
    print("  Optimising for F1 (tiebreak: precision)")

    margin_thresholds        = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    concentration_thresholds = [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]
    gap_at_thresholds        = [1, 2, 3, 99]   # 99 = effectively disable gap_at

    best, all_results = grid_search(
        features, margin_thresholds, concentration_thresholds, gap_at_thresholds
    )

    m, c, g, r = best
    gap_str = f">= {g}" if g < 99 else "disabled"
    print(f"\n  Best combo  : margin < {m}  OR  concentration < {c}  OR  gap_at {gap_str}")
    print(f"  Precision   : {r['precision']:.1%}")
    print(f"  Recall      : {r['recall']:.1%}")
    print(f"  F1          : {r['f1']:.3f}")
    print(f"  Flagged     : {r['flagged']}  (TP={r['tp']} FP={r['fp']})")
    print(f"  Missed      : {r['fn']}")
    if r["missed_ids"]:
        print(f"\n  Still-missed queries:")
        for qid, q in r["missed_ids"]:
            print(f"    [{qid}] {q}")

    # ── Top 10 by F1 from grid search ─────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  TOP 10 COMBOS BY F1")
    print(f"  {'margin':>8}  {'conc':>6}  {'gap_at':>8}  {'prec':>7}  {'recall':>7}  {'f1':>7}  {'flagged':>8}")
    sorted_results = sorted(all_results, key=lambda x: (-x[3]["f1"], -x[3]["precision"]))
    for m, c, g, r in sorted_results[:10]:
        gap_str = f">={g}" if g < 99 else "  off"
        print(f"  {m:>8}  {c:>6}  {gap_str:>8}  "
              f"{r['precision']:>6.1%}  {r['recall']:>6.1%}  "
              f"{r['f1']:>7.3f}  {r['flagged']:>8}")

    print()


if __name__ == "__main__":
    main()