import json
import random
from pathlib import Path
from collections import defaultdict

def main():
    base_dir = Path(__file__).parent
    input_path = base_dir / "Spider_extracted" / "extracted_questions.json"
    train_path = base_dir / "train_queries.json"
    test_path = base_dir / "test_queries.json"

    with open(input_path, "r") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions")

    # Group questions by db_id
    by_db = defaultdict(list)
    for q in questions:
        by_db[q["db_id"]].append(q)

    print(f"Found {len(by_db)} databases")

    random.seed(42)

    train = []
    test = []

    for db_id in sorted(by_db):
        db_questions = by_db[db_id]
        random.shuffle(db_questions)
        mid = len(db_questions) // 2
        train.extend(db_questions[:mid])
        test.extend(db_questions[mid:])

    # Verify no overlap
    train_ids = {q["question_id"] for q in train}
    test_ids = {q["question_id"] for q in test}
    assert train_ids.isdisjoint(test_ids), "Overlap detected between train and test!"

    with open(train_path, "w") as f:
        json.dump(train, f, indent=2)

    with open(test_path, "w") as f:
        json.dump(test, f, indent=2)

    # Build per-database breakdown
    train_by_db = defaultdict(int)
    test_by_db = defaultdict(int)
    for q in train:
        train_by_db[q["db_id"]] += 1
    for q in test:
        test_by_db[q["db_id"]] += 1

    per_db = {}
    for db_id in sorted(by_db):
        per_db[db_id] = {
            "total": len(by_db[db_id]),
            "train": train_by_db[db_id],
            "test": test_by_db[db_id],
        }

    metadata = {
        "total_questions": len(questions),
        "train_questions": len(train),
        "test_questions": len(test),
        "total_databases": len(by_db),
        "split_ratio": "50/50 per database",
        "random_seed": 42,
        "per_database": per_db,
    }

    metadata_path = base_dir / "train_test_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Train: {len(train)} questions -> {train_path}")
    print(f"Test:  {len(test)} questions -> {test_path}")
    print(f"Total: {len(train) + len(test)} (no overlap)")
    print(f"Metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
