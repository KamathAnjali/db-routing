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
    rank = entry.get('correct_db_rank')
    
    # Label: 1 if LLM was wrong (Ambiguous), 0 if it was right (Clear)
    label = 1 if rank is None or rank > 1 else 0

    question_text = entry.get('question', "")
    reasoning = entry.get('reasoning', {})
    
    # --- FEATURE ENGINEERING ---
    # Feature A: Score Decay (Gap between Top 1 and Top 3)
    top3_score = entry['top10'][2]['llm_score'] if len(entry['top10']) > 2 else 0
    decay = top1_score - top3_score
    
    # Feature B: Reasoning Conflict (Did it pick a DB it said it eliminated?)
    eliminated = reasoning.get('step2_eliminated', [])
    logic_clash = 1 if top1_id in eliminated else 0
    
    # Feature C: Lexical Complexity (Question length and SQL hints)
    q_len = len(question_text.split())
    has_agg = 1 if any(word in question_text.lower() for word in ['average', 'total', 'each', 'count', 'max', 'min']) else 0

    features = {
        'concentration': top1_score,
        'margin': top1_score - top2_score,
        'entropy': entry.get('overall_entropy', 0),
        'decay': decay,
        'logic_clash': logic_clash,
        'q_len': q_len,
        'has_agg': has_agg,
        'actual_rank': rank,
        'is_ambiguous': label
    }
    extracted_data.append(features)

df = pd.DataFrame(extracted_data)

# 2. SPLIT DATA (80% Train, 20% Test)
# We keep the indices to map back to the original question text later
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Define feature columns clearly
feature_cols = ['concentration', 'margin', 'entropy', 'decay', 'logic_clash', 'q_len', 'has_agg']

X_train = df_train[feature_cols]
y_train = df_train['is_ambiguous']

X_test = df_test[feature_cols]
y_test = df_test['is_ambiguous']

# 3. TRAIN DECISION TREE
clf = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# 4. PREDICT AND EVALUATE
df_test['prediction'] = clf.predict(X_test)

print("--- Classifier Performance ---")
print(confusion_matrix(y_test, df_test['prediction']))
print(classification_report(y_test, df_test['prediction']))

# 5. CALCULATE THEORETICAL UPPER BOUND
total_test = len(df_test)
rank_1_correct = len(df_test[df_test['actual_rank'] == 1])

# Fixable: Correct DB was in Top 10 (Rank > 1) AND we flagged it
fixable_and_flagged = len(df_test[(df_test['actual_rank'] > 1) & (df_test['prediction'] == 1)])

# Unfixable: Correct DB wasn't in Top 10 at all
not_in_top10 = df_test['actual_rank'].isna().sum()

absolute_ceiling = ((total_test - not_in_top10) / total_test) * 100
current_upper_bound = ((rank_1_correct + fixable_and_flagged) / total_test) * 100

print("\n--- Theoretical Upper Bounds (80/20 Split) ---")
print(f"Total Test Samples: {total_test}")
print(f"Absolute Ceiling (Retrieval Limit): {absolute_ceiling:.2f}%")
print(f"Current Upper Bound (System Limit): {current_upper_bound:.2f}%")
print(f"Unfixable Queries (Not in Top 10): {not_in_top10}")

# 6. VISUALIZE THE TREE
plt.figure(figsize=(20,10))
plot_tree(clf, 
          feature_names=feature_cols, 
          class_names=['Clear', 'Ambiguous'], 
          filled=True, 
          rounded=True)
plt.title("Decision Tree Flowchart (With Content Features)")
plt.show()

# --- SAVING THE AMBIGUOUS QUERIES ---
ambiguous_df = df_test[df_test['prediction'] == 1]
output_list = []

for index, row in ambiguous_df.iterrows():
    original_entry = data[index] 
    output_list.append({
        "question_id": original_entry.get("question_id"),
        "question": original_entry.get("question"),
        "top1_db": original_entry['top10'][0]['db_id'],
        "correct_db": original_entry.get("correct_db"),
        "predicted_rank": "Not in Top 10" if pd.isna(row['actual_rank']) else int(row['actual_rank']),   
        "confidence_scores": {
            "margin": float(row['margin']),
            "entropy": float(row['entropy']),
            "decay": float(row['decay'])
        },
        "logic_clash": int(row['logic_clash'])
    })

with open('flagged_ambiguous_queries.json', 'w') as f:
    json.dump(output_list, f, indent=4)

print(f"\nSuccessfully saved {len(output_list)} flagged queries.")