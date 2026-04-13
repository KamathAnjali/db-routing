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
    # Basic safety check for data integrity
    if not entry.get('top10') or len(entry['top10']) < 2:
        continue
        
    top1_score = entry['top10'][0]['llm_score']
    top2_score = entry['top10'][1]['llm_score']
    rank = entry.get('correct_db_rank')
    
    # Define our Label: 1 if LLM was wrong (Ambiguous), 0 if it was right (Clear)
    # We treat 'None' as wrong because the correct DB wasn't found at all
    label = 1 if rank is None or rank > 1 else 0

    features = {
        'concentration': top1_score,
        'margin': top1_score - top2_score,
        'entropy': entry.get('overall_entropy', 0),
        'is_ambiguous': label,
        'actual_rank': rank # Stored for the final bound calculation
    }
    extracted_data.append(features)

df = pd.DataFrame(extracted_data)

# 2. SPLIT DATA (80% Train, 20% Test)
# We use df_test later to access the 'actual_rank' for the final math
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

X_train = df_train[['concentration', 'margin', 'entropy']]
y_train = df_train['is_ambiguous']

X_test = df_test[['concentration', 'margin', 'entropy']]
y_test = df_test['is_ambiguous']

# 3. TRAIN DECISION TREE
# 'class_weight' makes the model more paranoid about missing errors (False Negatives)
# 'max_depth' keeps the flowchart simple and readable
clf = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
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

# 6. VISUALIZE THE FLOWCHART
plt.figure(figsize=(20,10))
plot_tree(clf, 
          feature_names=['Concentration', 'Margin', 'Entropy'], 
          class_names=['Clear', 'Ambiguous'], 
          filled=True, 
          rounded=True)
plt.title("How the System Decides to Flag a Database Query")
plt.show()

# --- SAVING THE AMBIGUOUS QUERIES ---

# 1. We filter the test dataframe for queries the model flagged as '1'
ambiguous_df = df_test[df_test['prediction'] == 1]

# 2. We can merge back with the original data to get the question text 
# (Since df_test only has features and IDs, we use the index to find the original text)
output_list = []
for index, row in ambiguous_df.iterrows():
    # 'index' in df matches the original order in our 'data' list
    original_entry = data[index] 
    
    # We create a record of what happened
    record = {
        "question_id": original_entry.get("question_id"),
        "question": original_entry.get("question"),
        "top1_db": original_entry['top10'][0]['db_id'],
        "correct_db": original_entry.get("correct_db"),
        "predicted_rank": "Not in Top 10" if pd.isna(row['actual_rank']) else row['actual_rank'],   
        "confidence_scores": {
            "margin": row['margin'],
            "entropy": row['entropy'],
            "concentration": row['concentration'],
        }
    }
    output_list.append(record)

# 3. Save to a new JSON file
with open('flagged_ambiguous_queries.json', 'w') as f:
    json.dump(output_list, f, indent=4)

print(f"\nSuccessfully saved {len(output_list)} ambiguous queries to 'flagged_ambiguous_queries.json'")

# --- DEEP DIVE INTO TOP 10 FAILURES ---

# 1. Analyze the 28 (False Negatives): Classifier said 'Clear' (0), but LLM was 'Wrong' (1)
false_negatives = df_test[(df_test['prediction'] == 0) & (df_test['is_ambiguous'] == 1)]
fn_not_in_top10 = false_negatives['actual_rank'].isna().sum()

# 2. Analyze the 97 (True Positives): Classifier said 'Ambiguous' (1), and LLM was 'Wrong' (1)
true_positives = df_test[(df_test['prediction'] == 1) & (df_test['is_ambiguous'] == 1)]
tp_not_in_top10 = true_positives['actual_rank'].isna().sum()

print("\n--- Failure Analysis: Retrieval vs. Classification ---")
print(f"Out of the 28 False Negatives: {fn_not_in_top10} were not in Top 10 (Impossible to fix)")
print(f"Out of the 28 False Negatives: {28 - fn_not_in_top10} were in Top 10 (Classifier simply missed them)")

print(f"\nOut of the 97 True Positives: {tp_not_in_top10} were not in Top 10 (Flagged correctly as 'Impossible')")
print(f"Out of the 97 True Positives: {97 - tp_not_in_top10} were in Top 10 (Flagged correctly as 'Fixable')")
