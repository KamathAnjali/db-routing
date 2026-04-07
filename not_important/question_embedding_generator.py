import sys
import json
import time
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("train", "test"):
        print("Usage: python question_embedding_generator.py <train|test>")
        sys.exit(1)

    split = sys.argv[1]

    base_dir = Path(__file__).parent
    input_path = base_dir / f"{split}_queries.json"
    output_dir = base_dir / "question_embeddings" / split

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions from {input_path.name}")

    # Expected embedding shape for Qwen3-Embedding-4B
    EXPECTED_SHAPE = (2560,)

    # Check existing embeddings and filter out those with correct shape
    def needs_embedding(q):
        q_id = str(q["question_id"])
        emb_path = output_dir / f"{q_id}.npy"
        if not emb_path.exists():
            return True
        try:
            emb = np.load(emb_path)
            if emb.shape != EXPECTED_SHAPE:
                print(f"  Invalid shape for {q_id}.npy: {emb.shape}, will regenerate")
                return True
            return False
        except Exception as e:
            print(f"  Error loading {q_id}.npy: {e}, will regenerate")
            return True

    print("Checking existing embeddings...")
    questions_to_process = [q for q in questions if needs_embedding(q)]

    skipped_count = len(questions) - len(questions_to_process)
    if skipped_count > 0:
        print(f"Skipping {skipped_count} questions with valid embeddings (shape={EXPECTED_SHAPE})")

    if not questions_to_process:
        print("All embeddings already exist with correct shape. Nothing to do.")
        return

    print(f"Processing {len(questions_to_process)} questions (missing or invalid shape)\n")

    # Same model and settings as schema_embedding_generator.py
    print("Loading Qwen3-Embedding-4B model (CPU)...")
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-4B",
        device="cpu",
        tokenizer_kwargs={"padding_side": "left"},
    )
    print("Model loaded.\n")

    start_time = time.time()
    total_to_process = len(questions_to_process)

    for i, q in enumerate(questions_to_process, 1):
        q_id = q["question_id"]
        text = q["question"].strip()

        if not text:
            print(f"[{i}/{total_to_process}] SKIPPED (empty): q_id={q_id}")
            continue

        # Custom task instruction for DB routing (Qwen3 docs recommend tailored
        # instructions over the generic "query" prompt for 1-5% improvement)
        embedding = model.encode(
            text,
            prompt="Instruct: Given a natural language question about a database, retrieve the database schema that the question should be routed to\nQuery: ",
            normalize_embeddings=True,
        )

        out_path = output_dir / f"{q_id}.npy"
        np.save(out_path, embedding)

        print(f"[{i}/{total_to_process}] Saved {q_id}.npy  (dim={embedding.shape[0]})")

    elapsed = time.time() - start_time
    print(f"\nDone — {total_to_process} new embeddings generated in {elapsed:.1f}s")
    print(f"Total embeddings now: {len(questions)} ({skipped_count} existed + {total_to_process} new)")
    print(f"Embeddings saved to {output_dir}/")


if __name__ == "__main__":
    main()
