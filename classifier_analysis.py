"""
Reranking Failure Classifier & Disambiguator Flagging
------------------------------------------------------
Pipeline:
  1. Ingest your softmax distributions + gold labels
  2. Extract features (entropy, margin, variance, etc.)
  3. Fit Logistic Regression → inspect coefficients
  4. Fit shallow Decision Tree → extract human-readable rules
  5. Plot PR curve → pick your flagging threshold
  6. Evaluate: % flagged, % failures caught, % false alarms

INPUT FORMAT (expected):
  A list of dicts, one per query:
  {
    "query_id": str,
    "scores": [float] * 10,   # raw LLM scores for top-10 DBs, in rank order
    "gold_rank": int,          # 0-indexed position of gold DB in top-10 (None if not in top-10)
    "query": str,              # optional, for inspection
    "db_ids": [str] * 10,      # optional, DB identifiers in rank order
  }
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    classification_report, roc_auc_score
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATOR (replace with yours)
# ─────────────────────────────────────────────

def generate_synthetic_data(n=500, seed=42):
    """
    Replace this with your actual data loader.
    Returns list of dicts matching the INPUT FORMAT above.
    """
    rng = np.random.default_rng(seed)
    data = []

    for i in range(n):
        # Simulate three failure modes:
        mode = rng.choice(["correct", "rerank_fail_ambiguous", "rerank_fail_confident"], p=[0.78, 0.12, 0.10])

        if mode == "correct":
            # Peaked distribution, gold is rank 0
            raw = rng.exponential([3.0] + [0.5]*9)
            gold_rank = 0

        elif mode == "rerank_fail_ambiguous":
            # Flat distribution — model is uncertain, gold is buried
            raw = rng.exponential([1.0]*10)
            gold_rank = int(rng.integers(1, 10))

        else:  # confident_wrong
            # Peaked but on wrong DB — hardest case
            raw = rng.exponential([3.0] + [0.5]*9)
            gold_rank = int(rng.integers(1, 5))

        # Softmax
        scores = np.exp(raw) / np.exp(raw).sum()

        data.append({
            "query_id": f"q{i:04d}",
            "scores": scores.tolist(),
            "gold_rank": gold_rank,
        })

    return data


# ─────────────────────────────────────────────
# 2. FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_features(record):
    """
    Extract all distribution-based features from one record.
    These are the signals available at inference time (no gold needed).
    """
    p = np.array(record["scores"])
    p = np.clip(p, 1e-10, 1.0)
    p = p / p.sum()  # ensure sums to 1

    sorted_p = np.sort(p)[::-1]

    feats = {
        # Uncertainty
        "entropy":          scipy_entropy(p),                    # overall spread
        "entropy_norm":     scipy_entropy(p) / np.log(len(p)),  # normalized 0-1

        # Top-1 confidence
        "top1_score":       sorted_p[0],

        # Margin signals — most important for reranking
        "margin_1_2":       sorted_p[0] - sorted_p[1],          # top vs runner-up
        "margin_1_3":       sorted_p[0] - sorted_p[2],          # top vs 3rd
        "margin_2_3":       sorted_p[1] - sorted_p[2],          # runner-up gap

        # Spread
        "score_variance":   np.var(p),
        "score_std":        np.std(p),
        "top3_mass":        sorted_p[:3].sum(),                  # concentration in top-3
        "top1_top3_ratio":  sorted_p[0] / sorted_p[:3].sum(),   # dominance of winner

        # Tail behavior
        "bottom5_mass":     sorted_p[5:].sum(),                  # mass in tail
        "gini":             _gini(sorted_p),                     # inequality of distribution
    }
    return feats


def _gini(sorted_scores_desc):
    """Gini coefficient — 0=perfectly equal, 1=all mass on one."""
    n = len(sorted_scores_desc)
    s = sorted_scores_desc[::-1]  # ascending for gini formula
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * s) / (n * np.sum(s))) - (n + 1) / n


def build_feature_matrix(data, only_in_top10=True):
    """
    Build X (features) and y (label) from raw data.

    Label: rerank_failed = gold was in top-10 but NOT at rank 0
    Optionally filter out retrieval failures (gold not in top-10).
    """
    records = []
    for rec in data:
        gold_rank = rec.get("gold_rank")

        # Skip retrieval failures if requested
        if only_in_top10 and (gold_rank is None or gold_rank >= 10):
            continue

        feats = extract_features(rec)
        feats["query_id"] = rec["query_id"]

        # Label: 1 = reranking failed, 0 = correct
        feats["rerank_failed"] = int(gold_rank != 0) if gold_rank is not None else 1

        # Gold-side signals (only available with gold labels, for analysis)
        if gold_rank is not None:
            p = np.array(rec["scores"])
            p = p / p.sum()
            feats["gold_rank"] = gold_rank
            feats["gold_score"] = p[gold_rank] if gold_rank < len(p) else 0.0
            feats["score_delta"] = p[0] - (p[gold_rank] if gold_rank < len(p) else 0.0)
        else:
            feats["gold_rank"] = -1
            feats["gold_score"] = 0.0
            feats["score_delta"] = 1.0

        records.append(feats)

    df = pd.DataFrame(records)
    return df


FEATURE_COLS = [
    "entropy", "entropy_norm",
    "top1_score",
    "margin_1_2", "margin_1_3", "margin_2_3",
    "score_variance", "score_std",
    "top3_mass", "top1_top3_ratio",
    "bottom5_mass", "gini",
]


# ─────────────────────────────────────────────
# 3. CLASSIFIERS
# ─────────────────────────────────────────────

def fit_logistic_regression(X, y):
    """Logistic regression with CV — returns fitted pipeline + OOF probabilities."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

    pipe.fit(X, y)  # fit on full data for inspection
    return pipe, oof_probs


def fit_decision_tree(X, y, feature_names, max_depth=4):
    """Shallow decision tree — returns fitted model + OOF probabilities."""
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        min_samples_leaf=10,
        random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

    clf.fit(X, y)
    rules = export_text(clf, feature_names=feature_names, max_depth=max_depth)
    return clf, oof_probs, rules


def get_lr_coefficients(pipe, feature_names):
    """Extract and sort LR coefficients by magnitude."""
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]
    # Scale coefficients back to original feature units
    coefs = clf.coef_[0] * scaler.scale_
    df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    return df.sort_values("coefficient", key=abs, ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# 4. THRESHOLD ANALYSIS
# ─────────────────────────────────────────────

def analyze_threshold(y_true, y_prob, flag_cost_ratio=2.0):
    """
    Compute precision, recall, F-score at every threshold.
    flag_cost_ratio: how many false alarms you'll tolerate per true catch.
    Returns threshold analysis dataframe + recommended threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # F-beta: beta > 1 weights recall more (catch more failures)
    beta = flag_cost_ratio
    f_beta = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall + 1e-10)

    rows = []
    for i, thresh in enumerate(thresholds):
        preds = (y_prob >= thresh).astype(int)
        flagged_pct = preds.mean() * 100
        rows.append({
            "threshold": thresh,
            "precision": precision[i],
            "recall": recall[i],
            "f_beta": f_beta[i],
            "flagged_pct": flagged_pct,
            "failures_caught": recall[i] * 100,
            "false_alarm_rate": (1 - precision[i]) * 100,
        })

    df = pd.DataFrame(rows)
    best_idx = df["f_beta"].idxmax()
    recommended_threshold = df.loc[best_idx, "threshold"]

    return df, recommended_threshold


def evaluate_at_threshold(y_true, y_prob, threshold):
    """Print a clean summary at a chosen threshold."""
    preds = (y_prob >= threshold).astype(int)
    total = len(y_true)
    flagged = preds.sum()
    failures = y_true.sum()
    caught = ((preds == 1) & (y_true == 1)).sum()
    false_alarms = ((preds == 1) & (y_true == 0)).sum()

    print(f"\n{'='*50}")
    print(f"  EVALUATION @ threshold = {threshold:.3f}")
    print(f"{'='*50}")
    print(f"  Total queries:        {total}")
    print(f"  Actual failures:      {failures} ({failures/total*100:.1f}%)")
    print(f"  Flagged for disambig: {flagged} ({flagged/total*100:.1f}%)")
    print(f"  Failures caught:      {caught} / {failures} ({caught/failures*100:.1f}% recall)")
    print(f"  False alarms:         {false_alarms} ({false_alarms/flagged*100:.1f}% of flagged)")
    print(f"  Passed through wrong: {failures - caught} queries will still fail")
    print(f"{'='*50}\n")
    print(classification_report(y_true, preds, target_names=["Pass", "Flag"]))


# ─────────────────────────────────────────────
# 5. INFERENCE-TIME FLAGGING (no gold needed)
# ─────────────────────────────────────────────

class RerankerFlagSystem:
    """
    Wraps the trained classifier for inference.
    At inference time: you have scores[] but no gold_rank.
    """
    def __init__(self, lr_pipe, threshold):
        self.pipe = lr_pipe
        self.threshold = threshold

    def flag(self, scores):
        """
        scores: list of 10 floats (softmax distribution over top-10 DBs)
        Returns: (should_flag: bool, failure_prob: float, signals: dict)
        """
        record = {"scores": scores, "gold_rank": 0, "query_id": "inf"}
        feats = extract_features(record)
        X = np.array([[feats[f] for f in FEATURE_COLS]])
        prob = self.pipe.predict_proba(X)[0, 1]
        return prob >= self.threshold, prob, feats

    def flag_batch(self, score_matrix):
        """score_matrix: shape (N, 10)"""
        records = [{"scores": s, "gold_rank": 0, "query_id": str(i)}
                   for i, s in enumerate(score_matrix)]
        X = np.array([[extract_features(r)[f] for f in FEATURE_COLS] for r in records])
        probs = self.pipe.predict_proba(X)[:, 1]
        flags = probs >= self.threshold
        return flags, probs


# ─────────────────────────────────────────────
# 6. PLOTTING
# ─────────────────────────────────────────────

def plot_full_analysis(df, y_true, lr_probs, dt_probs, coef_df, dt_rules, thresh_df, recommended_thresh):
    fig = plt.figure(figsize=(20, 16), facecolor="#0d1117")
    fig.suptitle("Reranker Failure Analysis", fontsize=22, color="#e6edf3",
                 fontname="monospace", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    PASS_COLOR = "#3fb950"
    FAIL_COLOR = "#f85149"
    LR_COLOR = "#58a6ff"
    DT_COLOR = "#d2a8ff"
    GRID_COLOR = "#21262d"
    TEXT_COLOR = "#e6edf3"
    MUTED_COLOR = "#8b949e"

    def style_ax(ax, title):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors=MUTED_COLOR, labelsize=8)
        ax.spines[:].set_color(GRID_COLOR)
        ax.set_title(title, color=TEXT_COLOR, fontsize=10, pad=8, fontname="monospace")
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)
        ax.xaxis.label.set_color(MUTED_COLOR)
        ax.yaxis.label.set_color(MUTED_COLOR)

    # ── A: Score distribution by outcome
    ax_a = fig.add_subplot(gs[0, 0])
    passed = df[df["rerank_failed"] == 0]["top1_score"]
    failed = df[df["rerank_failed"] == 1]["top1_score"]
    bins = np.linspace(0, 1, 30)
    ax_a.hist(passed, bins=bins, alpha=0.7, color=PASS_COLOR, label="Correct", density=True)
    ax_a.hist(failed, bins=bins, alpha=0.7, color=FAIL_COLOR, label="Failed", density=True)
    ax_a.legend(fontsize=8, facecolor="#21262d", labelcolor=TEXT_COLOR)
    ax_a.set_xlabel("top1_score")
    style_ax(ax_a, "A · Top-1 Score by Outcome")

    # ── B: Margin distribution by outcome
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.hist(df[df["rerank_failed"] == 0]["margin_1_2"], bins=30, alpha=0.7,
              color=PASS_COLOR, label="Correct", density=True)
    ax_b.hist(df[df["rerank_failed"] == 1]["margin_1_2"], bins=30, alpha=0.7,
              color=FAIL_COLOR, label="Failed", density=True)
    ax_b.legend(fontsize=8, facecolor="#21262d", labelcolor=TEXT_COLOR)
    ax_b.set_xlabel("margin (p[0] - p[1])")
    style_ax(ax_b, "B · Margin by Outcome")

    # ── C: Entropy by outcome
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.hist(df[df["rerank_failed"] == 0]["entropy"], bins=30, alpha=0.7,
              color=PASS_COLOR, label="Correct", density=True)
    ax_c.hist(df[df["rerank_failed"] == 1]["entropy"], bins=30, alpha=0.7,
              color=FAIL_COLOR, label="Failed", density=True)
    ax_c.legend(fontsize=8, facecolor="#21262d", labelcolor=TEXT_COLOR)
    ax_c.set_xlabel("entropy")
    style_ax(ax_c, "C · Entropy by Outcome")

    # ── D: LR Coefficients
    ax_d = fig.add_subplot(gs[1, 0])
    top_coef = coef_df.head(10)
    colors = [PASS_COLOR if c < 0 else FAIL_COLOR for c in top_coef["coefficient"]]
    bars = ax_d.barh(top_coef["feature"][::-1], top_coef["coefficient"][::-1],
                     color=colors[::-1], edgecolor="none", height=0.6)
    ax_d.axvline(0, color=MUTED_COLOR, linewidth=0.8)
    ax_d.set_xlabel("coefficient (→ predicts failure)")
    style_ax(ax_d, "D · LR Feature Importance")

    # ── E: PR Curve — both models
    ax_e = fig.add_subplot(gs[1, 1])
    for probs, color, label in [
        (lr_probs, LR_COLOR, f"Logistic Reg (AP={average_precision_score(y_true, lr_probs):.2f})"),
        (dt_probs, DT_COLOR, f"Decision Tree (AP={average_precision_score(y_true, dt_probs):.2f})"),
    ]:
        p, r, _ = precision_recall_curve(y_true, probs)
        ax_e.plot(r, p, color=color, linewidth=2, label=label)
    ax_e.axhline(y_true.mean(), color=MUTED_COLOR, linestyle="--", linewidth=1, label="Baseline")
    ax_e.set_xlabel("Recall (failures caught)")
    ax_e.set_ylabel("Precision")
    ax_e.set_xlim([0, 1]); ax_e.set_ylim([0, 1])
    ax_e.legend(fontsize=7.5, facecolor="#21262d", labelcolor=TEXT_COLOR)
    style_ax(ax_e, "E · Precision-Recall Curve")

    # ── F: Threshold analysis — flagged% vs recall
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.plot(thresh_df["threshold"], thresh_df["flagged_pct"],
              color=LR_COLOR, linewidth=2, label="% Flagged")
    ax_f.plot(thresh_df["threshold"], thresh_df["failures_caught"],
              color=FAIL_COLOR, linewidth=2, label="% Failures Caught")
    ax_f.axvline(recommended_thresh, color="#e3b341", linewidth=1.5,
                 linestyle="--", label=f"Rec. threshold={recommended_thresh:.2f}")
    ax_f.set_xlabel("Decision Threshold")
    ax_f.set_ylabel("%")
    ax_f.legend(fontsize=8, facecolor="#21262d", labelcolor=TEXT_COLOR)
    style_ax(ax_f, "F · Threshold vs Coverage Tradeoff")

    # ── G: Gold rank distribution for failures
    ax_g = fig.add_subplot(gs[2, 0])
    fail_df = df[df["rerank_failed"] == 1]
    rank_counts = fail_df["gold_rank"].value_counts().sort_index()
    ax_g.bar(rank_counts.index, rank_counts.values, color=FAIL_COLOR,
             edgecolor="none", width=0.7)
    ax_g.set_xlabel("Gold DB rank (0=correct)")
    ax_g.set_ylabel("Count")
    style_ax(ax_g, "G · Where Gold Lands in Failures")

    # ── H: Failure prob scatter — margin vs entropy
    ax_h = fig.add_subplot(gs[2, 1])
    sc = ax_h.scatter(df["margin_1_2"], df["entropy"],
                      c=lr_probs, cmap="RdYlGn_r", s=18, alpha=0.7, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax_h, label="failure prob").ax.yaxis.label.set_color(MUTED_COLOR)
    ax_h.set_xlabel("margin_1_2")
    ax_h.set_ylabel("entropy")
    style_ax(ax_h, "H · Failure Prob Landscape")

    # ── I: Confident-wrong analysis
    ax_i = fig.add_subplot(gs[2, 2])
    conf_wrong = df[(df["rerank_failed"] == 1) & (df["top1_score"] > 0.5)]
    conf_right = df[(df["rerank_failed"] == 0) & (df["top1_score"] > 0.5)]
    ax_i.scatter(conf_wrong["top1_score"], conf_wrong["score_delta"],
                 color=FAIL_COLOR, alpha=0.6, s=20, label=f"Confident wrong (n={len(conf_wrong)})")
    ax_i.scatter(conf_right["top1_score"], [0]*len(conf_right),
                 color=PASS_COLOR, alpha=0.3, s=10, label=f"Confident right (n={len(conf_right)})")
    ax_i.set_xlabel("top1_score")
    ax_i.set_ylabel("score_delta (p[0] - p[gold])")
    ax_i.legend(fontsize=7.5, facecolor="#21262d", labelcolor=TEXT_COLOR)
    style_ax(ax_i, "I · Confident-Wrong Cases")

    import os
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rerank_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Plot saved → {out_path}")
    plt.close()


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def run_pipeline(data, flag_cost_ratio=2.0):
    """
    flag_cost_ratio: beta for F-beta score.
      > 1 = prioritize recall (catch more failures, tolerate more false alarms)
      = 1 = balanced precision/recall
      < 1 = prioritize precision (fewer false alarms, miss more failures)
    """
    print("Building feature matrix...")
    df = build_feature_matrix(data, only_in_top10=True)
    X = df[FEATURE_COLS].values
    y = df["rerank_failed"].values

    print(f"Dataset: {len(df)} queries | {y.sum()} failures ({y.mean()*100:.1f}%)")

    # ── Logistic Regression
    print("\nFitting Logistic Regression...")
    lr_pipe, lr_probs = fit_logistic_regression(X, y)
    coef_df = get_lr_coefficients(lr_pipe, FEATURE_COLS)

    print("\nTop LR Coefficients (positive = predicts failure):")
    print(coef_df.head(8).to_string(index=False))
    print(f"\nLR ROC-AUC: {roc_auc_score(y, lr_probs):.3f}")
    print(f"LR Avg Precision: {average_precision_score(y, lr_probs):.3f}")

    # ── Decision Tree
    print("\nFitting Decision Tree...")
    dt_clf, dt_probs, dt_rules = fit_decision_tree(X, y, FEATURE_COLS, max_depth=4)
    print(f"\nDT ROC-AUC: {roc_auc_score(y, dt_probs):.3f}")
    print(f"\nDecision Tree Rules:\n{dt_rules}")

    # ── Threshold Analysis (on LR — better calibrated)
    thresh_df, recommended_thresh = analyze_threshold(y, lr_probs, flag_cost_ratio)
    evaluate_at_threshold(y, lr_probs, recommended_thresh)

    # ── Plot
    print("Generating analysis plots...")
    plot_full_analysis(df, y, lr_probs, dt_probs, coef_df, dt_rules,
                       thresh_df, recommended_thresh)

    # ── Return inference system
    flag_system = RerankerFlagSystem(lr_pipe, recommended_thresh)
    return flag_system, df, thresh_df


# ─────────────────────────────────────────────
# 8. USAGE EXAMPLE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── REPLACE THIS with your actual data loader ──
    # Example:
    # import json
    # with open("your_results.json") as f:
    #     data = json.load(f)
    #
    # Each record must have:
    #   "query_id": str
    #   "scores":   list of 10 floats (softmax distribution)
    #   "gold_rank": int (0-indexed, 0=correct, None if not in top-10)
    data = generate_synthetic_data(n=800)

    # ── Run analysis ──
    flag_system, df, thresh_df = run_pipeline(data, flag_cost_ratio=2.0)

    # ── Inference example ──
    print("\n── Inference Example ──")
    test_scores = [0.45, 0.42, 0.05, 0.03, 0.02, 0.01, 0.01, 0.005, 0.005, 0.0]
    should_flag, prob, signals = flag_system.flag(test_scores)
    print(f"Scores: {[round(s,3) for s in test_scores]}")
    print(f"Failure probability: {prob:.3f}")
    print(f"Flag for disambiguator: {should_flag}")
    print(f"Key signals: margin={signals['margin_1_2']:.3f}, entropy={signals['entropy']:.3f}")

    print("\nDone. Adjust flag_cost_ratio to tune precision/recall tradeoff.")
    print("  flag_cost_ratio=1  → balanced")
    print("  flag_cost_ratio=2  → catch more failures (default)")
    print("  flag_cost_ratio=0.5 → fewer false alarms")