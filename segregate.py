import json

KEEP_FIELDS = ['question_id', 'question', 'correct_db', 'correct_db_rank', 'top10', 'overall_entropy']

with open('llm_rerank_openrouter_reasoning.json', 'r') as f:
    data = json.load(f)

# Filter out null ranks
filtered = [item for item in data if item.get('correct_db_rank') is not None]

# Keep only specified fields
filtered = [{k: v for k, v in item.items() if k in KEEP_FIELDS} for item in filtered]

rank_1 = [item for item in filtered if item['correct_db_rank'] == 1]
not_rank_1 = [item for item in filtered if item['correct_db_rank'] != 1]

with open('rank_1.json', 'w') as f:
    json.dump(rank_1, f, indent=2)

with open('not_rank_1.json', 'w') as f:
    json.dump(not_rank_1, f, indent=2)

print(f"Rank 1: {len(rank_1)} items")
print(f"Not Rank 1: {len(not_rank_1)} items")
print(f"Excluded (null): {len(data) - len(filtered)} items")