import json
import time
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


def main():
    base_dir = Path(__file__).parent
    schema_dir = base_dir / "Spider_extracted" / "only_DDL_combined"
    output_dir = base_dir / "schema_embeddings"

    if not schema_dir.exists():
        raise FileNotFoundError(f"Schema directory not found: {schema_dir}")

    output_dir.mkdir(exist_ok=True)

    schema_files = sorted(schema_dir.glob("*_schema.txt"))
    print(f"Found {len(schema_files)} schema files in {schema_dir}")

    if not schema_files:
        print("No schema files found. Exiting.")
        return

    print("Loading Qwen3-Embedding-4B model (CPU)...")
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-4B",
        device="cpu",
        tokenizer_kwargs={"padding_side": "left"},
    )
    print("Model loaded.\n")

    start_time = time.time()

    for i, schema_file in enumerate(schema_files, 1):
        name = schema_file.stem.replace("_schema", "")  # e.g. "concert_singer"
        text = schema_file.read_text(encoding="utf-8").strip()

        if not text:
            print(f"[{i}/{len(schema_files)}] SKIPPED (empty): {name}")
            continue

        # Encode single schema — no instruction prefix needed for documents
        embedding = model.encode(text, normalize_embeddings=True)

        # Save as .npy with the same stem name as the source file
        out_path = output_dir / f"{name}.npy"
        np.save(out_path, embedding)

        print(f"[{i}/{len(schema_files)}] Saved {out_path.name}  (dim={embedding.shape[0]})")

    elapsed = time.time() - start_time
    print(f"\nDone — {len(schema_files)} schemas processed in {elapsed:.1f}s")
    print(f"Embeddings saved to {output_dir}/")


if __name__ == "__main__":
    main()