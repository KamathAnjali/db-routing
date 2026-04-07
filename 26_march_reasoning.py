"""
llm_rerank_openrouter.py
------------------------
Same pipeline as llm_rerank.py but uses OpenRouter API with comparative
chain-of-thought reasoning to prevent uniform score collapse.

TCS PROMPT USED + reasoning + random sampling

keep in mind this is configured only for train set

"""

import re
import json
import math
import time
import logging
from pathlib import Path
from openai import OpenAI
import os
import random


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME        = "deepseek/deepseek-chat-v3-0324"
MAX_RETRIES       = 3
RETRY_DELAY       = 5
REQUEST_DELAY     = 3.0
DEBUG_LIMIT       = 750    # set to None for full run

# ── Paths ─────────────────────────────────────────────────────────────────────
base_dir      = Path(__file__).parent
results_path  = base_dir / "retrieval_results/results_train_10.json"
schema_dir    = base_dir / "Spider_extracted" / "only_DDL_compressed"
output_path   = base_dir / "llm_rerank_openrouter_reasoning.json"

# ── OpenRouter client ─────────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_schema(db_id: str) -> str:
    path = schema_dir / f"{db_id}_compressed.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return f"[Schema not found for {db_id}]"


def shannon_entropy(scores: list[float]) -> float:
    entropy = 0.0
    for p in scores:
        if p > 0:
            entropy -= p * math.log(p)
    return round(entropy, 6)


def normalise(scores: list[float]) -> list[float]:
    total = sum(scores)
    if total == 0:
        n = len(scores)
        return [1.0 / n] * n
    return [s / total for s in scores]


def build_prompt(question: str, candidates: list[dict]) -> str:
    db_blocks = ""
    for i, c in enumerate(candidates, 1):
        db_blocks += f"\n--- Database {i}: {c['db_id']} ---\n{c['ddl']}\n"

    db_id_list = [c['db_id'] for c in candidates]

    return f"""
You are an expert DB Administrator tasked with evaluating and refining the relevance ranking of DBs for specific questions. You will be provided with a question and a list of 10 candidate DBs along with their schemas.

Your task is to analyze the question, critically assess the initial ranking, and provide a revised ranking of all 10 DBs. You must also assign a Confidence Score (0 to 1) to each, representing how accurately that database's schema can fulfill the specific query requirements.

## Question
"{question}"

## Candidate Database Schemas
{db_blocks}

## Rules: 
1. **Ranking**: Rank 1 must be the *undisputed* best match based on schema coverage.
2. **Selection**: All 10 DBs must be distinct and selected from the provided list.
3. **Confidence Scores**: Assign a decimal score to each of the 10 DBs. **The sum of all 10 scores must equal exactly 1.0.**

## Output Format
Respond with ONLY the following JSON. No markdown, no text outside the JSON.

{{
  "step0_domain": "<question domain and per-database domain classification>",
  "step1_requirements": "<specific tables and columns the question needs>",
  "step2_eliminated": ["<db_id>", "..."],
  "step3_comparison": "<your column-by-column comparison with explicit ✓/✗ evidence>",
  "scores": [
    {{"db_id": "{db_id_list[0]}", "score": <float>}},
    {{"db_id": "{db_id_list[1]}", "score": <float>}},
    {{"db_id": "{db_id_list[2]}", "score": <float>}},
    # ... (the LLM will follow the pattern for all 10)
    {{"db_id": "{db_id_list[9]}", "score": <float>}}
  ]
}}
"""


def call_llm(prompt: str, question_id: int) -> list[dict] | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise SQL database schema analyst. "
                            "You always classify domain first — databases from the wrong domain are eliminated immediately. "
                            "You always compare databases using specific column-level evidence, never vague impressions. "
                            "You never prefer a database because it has a simpler schema or fewer tables. "
                            "You always pick a clear winner — you never assign equal scores to databases "
                            "that you have compared head to head. "
                            "You respond with valid JSON only, exactly matching the requested format."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1400,
            )

            raw = response.choices[0].message.content.strip()

            # ── Strip markdown fences ─────────────────────────────────────────
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)

            parsed      = json.loads(raw)
            scores_list = parsed["scores"]

            if len(scores_list) != 10:
                raise ValueError(f"Expected 10 scores, got {len(scores_list)}")

            # ── Log the full reasoning chain ──────────────────────────────────
            log.info(f"  [q_id={question_id}] Domain       : {parsed.get('step0_domain', 'N/A')}")
            log.info(f"  [q_id={question_id}] Requirements : {parsed.get('step1_requirements', 'N/A')}")
            log.info(f"  [q_id={question_id}] Eliminated   : {parsed.get('step2_eliminated', [])}")
            log.info(f"  [q_id={question_id}] Comparison   : {parsed.get('step3_comparison', 'N/A')}")
            log.info(f"  [q_id={question_id}] Scores       : { {s['db_id']: s['score'] for s in scores_list} }")

            # ── Warn if uniform scores sneak through ──────────────────────────
            score_vals = [s["score"] for s in scores_list]
            non_zero   = [s for s in score_vals if s > 0]
            if len(set(round(s, 3) for s in non_zero)) == 1 and len(non_zero) > 1:
                log.warning(f"  [q_id={question_id}] ⚠️  Uniform scores detected despite comparative prompt.")

            return scores_list, parsed

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            log.warning(f"  [q_id={question_id}] Parse error attempt {attempt}: {e}")
            if 'raw' in locals():
                log.warning(f"  Raw was: {raw[:400]}")
        except Exception as e:
            log.warning(f"  [q_id={question_id}] API error attempt {attempt}: {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    log.error(f"  [q_id={question_id}] All {MAX_RETRIES} attempts failed.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load input ────────────────────────────────────────────────────────────
    log.info(f"Loading {results_path}")
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    queries = data["results"]

    random.seed(42)
    random.shuffle(queries) 

    # ── Apply debug limit ─────────────────────────────────────────────────────
    if DEBUG_LIMIT is not None:
        queries = queries[:DEBUG_LIMIT]
        log.info(f"DEBUG_LIMIT={DEBUG_LIMIT}: processing {len(queries)} queries only.")
    else:
        log.info(f"Total queries to process: {len(queries)}")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    output: list[dict] = []
    processed_ids: set[int] = set()

    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, list):
            output = [e for e in existing if "question_id" in e]
        else:
            output = []
        processed_ids = {entry["question_id"] for entry in output}
        log.info(f"Resuming — {len(processed_ids)} already done.")

    # ── Cost + time estimate ──────────────────────────────────────────────────
    remaining         = len(queries) - len(processed_ids)
    est_input_tokens  = remaining * 3000
    est_output_tokens = remaining * 700   # bumped slightly for longer reasoning
    est_cost = (est_input_tokens  / 1_000_000) * 0.27 + \
               (est_output_tokens / 1_000_000) * 1.10
    est_minutes = (remaining * REQUEST_DELAY) / 60
    log.info(
        f"Remaining: {remaining} queries | "
        f"Est. cost: ~${est_cost:.4f} | "
        f"Est. time: ~{est_minutes:.1f} min"
    )

    # ── Process each query ────────────────────────────────────────────────────
    start_time = time.time()

    for idx, entry in enumerate(queries, 1):
        q_id     = entry["question_id"]
        question = entry["question"]
        top10     = entry["top_10"]

        if q_id in processed_ids:
            log.info(f"[{idx}/{len(queries)}] q_id={q_id} already processed, skipping.")
            continue

        log.info(f"[{idx}/{len(queries)}] q_id={q_id}  \"{question[:60]}\"")

        # ── Build candidates with DDLs ────────────────────────────────────────
        candidates = [
            {"db_id": c["db_id"], "ddl": load_schema(c["db_id"])}
            for c in top10
        ]

        prompt     = build_prompt(question, candidates)
        result = call_llm(prompt, q_id)

        if result is None:
            scores_raw = None
            reasoning = None
        else:
            scores_raw, reasoning = result

        # ── Throttle ──────────────────────────────────────────────────────────
        time.sleep(REQUEST_DELAY)

        if scores_raw is None:
            scores_raw = [{"db_id": c["db_id"], "score": 1.0 / len(candidates)} for c in candidates]
            reasoning = {"error": "LLM failed after retries"}
            log.warning(f"  [q_id={q_id}] Using uniform fallback scores.")

        # ── Normalise ────────────────────────────(only if model doesnt sum up to 1 already)─────────────────────
        raw_floats  = [s["score"] for s in scores_raw]
        if not math.isclose(sum(raw_floats), 1.0, rel_tol=1e-3):
            log.warning("Scores don't sum to 1, normalizing.")
            norm_floats = normalise(raw_floats)
        else:
            norm_floats = raw_floats

        top10_out = [
            {"db_id": s["db_id"], "llm_score": round(norm_floats[i], 6)}
            for i, s in enumerate(scores_raw)
        ]

        entropy = shannon_entropy(norm_floats)

        # ── Compute correct_db_rank for easy accuracy tracking ────────────────
        top_10_out  = sorted(top10_out, key=lambda x: x["llm_score"], reverse=True)
        correct_db   = entry.get("correct_db")
        correct_rank = next(
            (i + 1 for i, x in enumerate(top_10_out) if x["db_id"] == correct_db),
            None
        )

        output.append({
            "question_id":     q_id,
            "question":        question,
            "correct_db":      correct_db,
            "correct_db_rank": correct_rank,
            "top10":            top_10_out,
            "overall_entropy": entropy,
            "reasoning":    reasoning,
        })

        # ── Log whether we got it right ───────────────────────────────────────
        rank_str = f"rank={correct_rank}" if correct_rank else "rank=NOT_FOUND"
        marker   = "✓" if correct_rank == 1 else "✗"
        log.info(f"  [q_id={q_id}] {marker} correct_db={correct_db} {rank_str}  entropy={entropy}")

        # ── ETA every 10 ─────────────────────────────────────────────────────
        done_so_far = len(output)
        if done_so_far % 10 == 0:
            elapsed     = time.time() - start_time
            rate        = elapsed / done_so_far
            remaining_n = remaining - done_so_far
            eta_min     = (remaining_n * rate) / 60
            correct_so_far = sum(1 for e in output if e["correct_db_rank"] == 1)
            log.info(
                f"  ── Progress: {done_so_far}/{remaining} done | "
                f"Top-1 acc: {correct_so_far}/{done_so_far} ({correct_so_far/done_so_far:.1%}) | "
                f"{rate:.1f}s/query | ETA: {eta_min:.1f} min"
            )

        # ── Checkpoint every 50 ───────────────────────────────────────────────
        if done_so_far % 50 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            log.info(f"  Checkpoint saved ({done_so_far} entries).")

    # ── Final save ────────────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    total_time = (time.time() - start_time) / 60

    # ── Final accuracy summary ────────────────────────────────────────────────
    total         = len(output)
    top1_correct  = sum(1 for e in output if e["correct_db_rank"] == 1)
    top2_correct  = sum(1 for e in output if e["correct_db_rank"] in (1, 2))
    not_found     = sum(1 for e in output if e["correct_db_rank"] is None)
    avg_entropy   = sum(e["overall_entropy"] for e in output) / total if total else 0

    log.info(f"\n{'='*55}")
    log.info(f"  FINAL SUMMARY — {total} queries")
    log.info(f"  Top-1 accuracy : {top1_correct}/{total} = {top1_correct/total:.1%}")
    log.info(f"  Top-2 accuracy : {top2_correct}/{total} = {top2_correct/total:.1%}")
    log.info(f"  Not in top10    : {not_found}")
    log.info(f"  Avg entropy    : {avg_entropy:.4f}")
    log.info(f"  Total time     : {total_time:.1f} min")
    log.info(f"  Saved to       : {output_path}")
    log.info(f"{'='*55}")


if __name__ == "__main__":
    main()