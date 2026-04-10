import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# 1. LOAD AND PREPARE DATA
file_path = 'llm_rerank_openrouter_reasoning.json'

with open(file_path, 'r') as f:
    data = json.load(f)

extracted_data = []
for entry in data:
    if not entry.get('top10') or len(entry['top10']) < 2:
        continue
        
    top1_score = entry['top10'][0]['llm_score']
    top2_score = entry['top10'][1]['llm_score']
    top1_id = entry['top10'][0]['db_id']
    
    # --- SELF-REFLECTION FEATURES ---
    reasoning = entry.get('reasoning', {})
    eliminated = reasoning.get('step2_eliminated', [])
    # Safe check for string conversion
    comparison = str(reasoning.get('step3_comparison', ""))
    
    # Logical Conflict: Did it pick a DB it said it eliminated?
    logical_conflict = 1 if top1_id in eliminated else 0
    
    # Hedge Count: Keywords indicating uncertainty in text
    uncertainty_keywords = ['but no', 'not explicit', 'not clear', 'no explicit', 'however']
    hedge_count = sum(1 for word in uncertainty_keywords if word in comparison.lower())
    
    # Step 3 Complexity: Number of database underscores mentioned
    comp_complexity = comparison.count('_')

    # Labels and Metadata
    rank = entry.get('correct_db_rank')
    label = 1 if rank is None or rank > 1 else 0

    extracted_data.append({
        'concentration': top1_score,
        'margin': top1_score - top2_score,
        'entropy': entry.get('overall_entropy', 0),
        'logical_conflict': logical_conflict,
        'hedge_count': hedge_count,
        'comp_complexity': comp_complexity,
        'is_ambiguous': label,
        'actual_rank': rank
    })

df = pd.DataFrame(extracted_data)

# 2. SPLIT DATA
feature_cols = ['concentration', 'margin', 'entropy', 'logical_conflict', 'hedge_count', 'comp_complexity']
X = df[feature_cols]
y = df['is_ambiguous']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAIN RANDOM FOREST
# Using 100 trees with balanced weights to prioritize catching errors
rf_clf = RandomForestClassifier(
    n_estimators=100, 
    max_depth=4, 
    class_weight='balanced', 
    random_state=42
)
rf_clf.fit(X_train, y_train)

# 4. PREDICT AND EVALUATE
y_pred = rf_clf.predict(X_test)

print("--- Random Forest Classifier Performance ---")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 5. FEATURE IMPORTANCE (The "Why")
print("\n--- Feature Importance ---")
importances = rf_clf.feature_importances_
for name, val in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
    print(f"{name}: {val:.4f}")

# 6. CALCULATE THEORETICAL UPPER BOUND
# Access the original 'actual_rank' for the test set
test_indices = X_test.index
actual_ranks_test = df.loc[test_indices, 'actual_rank']

total_test = len(y_test)
rank_1_correct = len(actual_ranks_test[actual_ranks_test == 1])

# Fixable: Rank 2-10 AND Predicted as Ambiguous (1)
fixable_mask = (actual_ranks_test > 1) & (y_pred == 1)
fixable_and_flagged = len(actual_ranks_test[fixable_mask])

not_in_top10 = actual_ranks_test.isna().sum()

absolute_ceiling = ((total_test - not_in_top10) / total_test) * 100
current_upper_bound = ((rank_1_correct + fixable_and_flagged) / total_test) * 100

print("\n--- Final Theoretical Upper Bounds ---")
print(f"Total Test Samples: {total_test}")
print(f"Absolute Ceiling (Retrieval Limit): {absolute_ceiling:.2f}%")
print(f"Current Upper Bound (System Limit): {current_upper_bound:.2f}%")
print(f"Unfixable Queries (Not in Top 10): {not_in_top10}")

# --- SAVING THE FLAG LIST ---
flagged_indices = test_indices[y_pred == 1]
output_list = []
for idx in flagged_indices:
    original = data[idx]
    output_list.append({
        "question": original.get("question"),
        "top1_db": original['top10'][0]['db_id'],
        "correct_db": original.get("correct_db"),
        "rank": original.get("correct_db_rank")
    })

with open('rf_flagged_queries.json', 'w') as f:
    json.dump(output_list, f, indent=4)