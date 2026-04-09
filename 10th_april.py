import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Load the JSON
with open('llm_rerank_openrouter_reasoning.json', 'r') as f:
    data = json.load(f)

extracted_data = []
for entry in data:
    if not entry.get('top10') or len(entry['top10']) < 2:
        continue
        
    top1_score = entry['top10'][0]['llm_score']
    top2_score = entry['top10'][1]['llm_score']
    rank = entry.get('correct_db_rank')
    
    # Label is 1 if LLM was wrong (including if rank is None)
    label = 1 if rank is None or rank > 1 else 0

    features = {
        'concentration': top1_score,
        'margin': top1_score - top2_score,
        'entropy': entry.get('overall_entropy', 0),
        'is_ambiguous': label,
        'actual_rank': rank # We keep this for the final math
    }
    extracted_data.append(features)

df = pd.DataFrame(extracted_data)

# 2. Split (Using the same random_state to keep results consistent)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# 3. Train
X_train = df_train[['concentration', 'margin', 'entropy']]
y_train = df_train['is_ambiguous']
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Predict on Test Set
X_test = df_test[['concentration', 'margin', 'entropy']]
df_test['prediction'] = model.predict(X_test)

# --- THE FINAL MATH ---

total_test = len(df_test)

# A: Correct DB was at Rank 1 (Already correct)
rank_1 = len(df_test[df_test['actual_rank'] == 1])

# B: Correct DB was at Rank 2-10 AND we flagged it (Fixed by Disambiguator)
fixed_by_disambiguator = len(df_test[
    (df_test['actual_rank'] > 1) & (df_test['prediction'] == 1)
])

# C: Correct DB was not in Top 10 at all (Unfixable)
not_in_top10 = len(df_test[df_test['actual_rank'].isna()])

# CALCULATIONS
absolute_ceiling = ((total_test - not_in_top10) / total_test) * 100
current_upper_bound = ((rank_1 + fixed_by_disambiguator) / total_test) * 100

print(f"Total Test Queries: {total_test}")
print(f"1. Absolute Ceiling (Retrieval Limit): {absolute_ceiling:.2f}%")
print(f"2. Current Upper Bound (System Limit): {current_upper_bound:.2f}%")
print(f"---")
print(f"Queries at Rank 1: {rank_1}")
print(f"Queries fixed from Rank 2-10: {fixed_by_disambiguator}")
print(f"Queries lost (Not in Top 10): {not_in_top10}")