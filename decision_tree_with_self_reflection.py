import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
    comparison = str(reasoning.get('step3_comparison', ""))
    
    # 1. Logical Conflict: Did it pick a DB it said it eliminated?
    logical_conflict = 1 if top1_id in eliminated else 0
    
    # 2. Hedge Count: Did it use words like "uncertain" or "not explicit"?
    uncertainty_keywords = ['but no', 'not explicit', 'not clear', 'no explicit', 'however']
    hedge_count = sum(1 for word in uncertainty_keywords if word in comparison.lower())
    
    # 3. Step 3 Complexity: How many schemas did it have to compare?
    # (Using the count of 'db' mentions or underscores as a proxy)
    comp_complexity = comparison.count('_')

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

# Update your X_train/X_test columns:
# X = df[['concentration', 'margin', 'entropy', 'logical_conflict', 'hedge_count', 'comp_complexity']]

df = pd.DataFrame(extracted_data)

# 2. SPLIT DATA (80% Train, 20% Test)
# We use df_test later to access the 'actual_rank' for the final math
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# --- UPDATE SECTION 2 ---
# Define the full set of features including Reflection
feature_cols = [
    'concentration', 
    'margin', 
    'entropy', 
    'logical_conflict', 
    'hedge_count', 
    'comp_complexity'
]

X_train = df_train[feature_cols]
y_train = df_train['is_ambiguous']

X_test = df_test[feature_cols]
y_test = df_test['is_ambiguous']

# 3. TRAIN DECISION TREE
# 'class_weight' makes the model more paranoid about missing errors (False Negatives)
# 'max_depth' keeps the flowchart simple and readable
clf = DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# 4. PREDICT AND EVALUATE
df_test['prediction'] = clf.predict(X_test)

print("--- Classifier Performance ---")
print(confusion_matrix(y_test, df_test['prediction']))
print(classification_report(y_test, df_test['prediction']))

# 5. CALCULATE THEORETICAL UPPER BOUND
total_test = len(df_test)

# Cases where the system ends up CORRECT:
# A: LLM was right at Rank 1
rank_1_correct = len(df_test[df_test['actual_rank'] == 1])

# B: LLM was wrong (Rank 2-10), but Classifier caught it (Flagged)
# We assume a perfect disambiguator fixes these
fixable_and_flagged = len(df_test[
    (df_test['actual_rank'] > 1) & (df_test['prediction'] == 1)
])

# C: Correct DB wasn't in the list at all (Absolute Loss)
not_in_top10 = len(df_test[df_test['actual_rank'].isna()])

# The Upper Limit Calculation
absolute_ceiling = ((total_test - not_in_top10) / total_test) * 100
current_upper_bound = ((rank_1_correct + fixable_and_flagged) / total_test) * 100

print("\n--- Theoretical Upper Bounds ---")
print(f"Total Test Samples: {total_test}")
print(f"Absolute Ceiling (Retrieval Limit): {absolute_ceiling:.2f}%")
print(f"Current Upper Bound (System Limit): {current_upper_bound:.2f}%")
print(f"Unfixable Queries (Not in Top 10): {not_in_top10}")

