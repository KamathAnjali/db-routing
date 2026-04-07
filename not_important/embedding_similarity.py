import sys
import json
import numpy as np
from pathlib import Path


def cosine_similarity(a, b):
    """Cosine similarity between vector a and matrix b (one row per db)."""
    dot = b @ a
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b, axis=1)
    return dot / (norm_a * norm_b + 1e-10)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("train", "test"):
        print("Usage: python embedding_similarity.py <train|test>")
        sys.exit(1)

    split = sys.argv[1]

    base_dir = Path(__file__).parent
    query_emb_dir = base_dir / "question_embeddings" / split
    schema_emb_dir = base_dir / "schema_embeddings"
    queries_path = base_dir / f"{split}_queries.json"
    output_path = base_dir / f"results_{split}_10.json"

    if not query_emb_dir.exists():
        raise FileNotFoundError(f"Question embeddings not found: {query_emb_dir}")
    if not schema_emb_dir.exists():
        raise FileNotFoundError(f"Schema embeddings not found: {schema_emb_dir}")
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")

    # Load queries
    with open(queries_path, "r") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} queries from {queries_path.name}")

    # Load all schema embeddings into a matrix
    schema_files = sorted(schema_emb_dir.glob("*.npy"))
    db_names = [f.stem for f in schema_files]
    schema_matrix = np.stack([np.load(f) for f in schema_files])  # (num_dbs, dim)
    print(f"Loaded {len(db_names)} schema embeddings (dim={schema_matrix.shape[1]})")

    # Process each question
    correct_at_1 = 0
    correct_at_5 = 0
    correct_at_10 = 0
    results = []

    for i, q in enumerate(questions, 1):
        q_id = q["question_id"]
        db_id = q["db_id"]
        question = q["question"]

        emb_path = query_emb_dir / f"{q_id}.npy"
        if not emb_path.exists():
            print(f"[{i}/{len(questions)}] MISSING embedding for q_id={q_id}, skipping")
            continue

        q_emb = np.load(emb_path)

        # Cosine similarity against all schemas
        scores = cosine_similarity(q_emb, schema_matrix)
    
        # Top 10
        top10_idx = np.argsort(scores)[::-1][:10]
        top10 = [
            {"db_id": db_names[idx], "score": round(float(scores[idx]), 6)}
            for idx in top10_idx
        ]

        rank = None
        for r, entry in enumerate(top10, 1):
            if entry["db_id"] == db_id:
                rank = r
                break

        if rank == 1:
            correct_at_1 += 1
        if rank is not None and rank <= 5:
            correct_at_5 += 1
        if rank is not None and rank <= 10:
            correct_at_10 += 1

        results.append({
            "question_id": q_id,
            "question": question,
            "correct_db": db_id,
            "correct_db_rank": rank,
            "top_10": top10,
        })

        if i % 500 == 0:
            print(f"  Processed {i}/{len(questions)}")

    total = len(results)
    recall_1 = correct_at_1 / total if total else 0
    recall_5 = correct_at_5 / total if total else 0
    recall_10 = correct_at_10 / total if total else 0

    output = {
        "split": split,
        "total_queries": total,
        "recall@1": round(recall_1, 4),
        "recall@5": round(recall_5, 4),
        "recall@10": round(recall_10, 4),
        "correct@1": correct_at_1,
        "correct@5": correct_at_5,
        "correct@10": correct_at_10,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*40}")
    print(f"Split:     {split}")
    print(f"Queries:   {total}")
    print(f"Recall@1:  {recall_1:.4f}  ({correct_at_1}/{total})")
    print(f"Recall@5:  {recall_5:.4f}  ({correct_at_5}/{total})")
    print(f"Recall@10:  {recall_10:.4f}  ({correct_at_10}/{total})")
    print(f"{'='*40}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()