# DB Routing

## Requirements

```
numpy
sentence-transformers
torch
```

Install into the venv:

```bash
source venv/bin/activate
pip install numpy sentence-transformers torch 
```

## Run Order

### 1. Split train/test queries

```bash
python train_test_query_split.py
```

No arguments. Reads `Spider_extracted/extracted_questions.json`, outputs `train_queries.json`, `test_queries.json`, and `train_test_metadata.json`.

### 2. Generate schema embeddings

```bash
python schema_embedding_generator.py
```

No arguments. Reads schemas from `Spider_extracted/only_DDL_combined/`, outputs `.npy` files to `schema_embeddings/`.

### 3. Generate question embeddings

```bash
python question_embedding_generator.py <train|test>
```

| Arg | Description |
|-----|-------------|
| `train` or `test` | Which split to embed |

Reads `train_queries.json` or `test_queries.json`, outputs `.npy` files to `question_embeddings/<split>/`.

### 4. Compute embedding similarity

```bash
python embedding_similarity.py <train|test>
```

| Arg | Description |
|-----|-------------|
| `train` or `test` | Which split to evaluate |

Computes cosine similarity between question and schema embeddings. Outputs `results_train.json` or `results_test.json` with Recall@1/3/5.