import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load the JSON file
with open('llm_rerank_openrouter_reasoning.json', 'r') as f:
    data = json.load(f)

# 2. Extract features and labels into a list
extracted_data = []

for entry in data:
    # 1. Get the top 2 scores safely
    # If top10 is empty for some reason, we skip this entry
    if not entry.get('top10') or len(entry['top10']) < 2:
        continue
        
    top1_score = entry['top10'][0]['llm_score']
    top2_score = entry['top10'][1]['llm_score']
    
    # 2. Get the rank safely
    rank = entry.get('correct_db_rank')
    
    # 3. Handle the NoneType error:
    # If rank is None, it means the correct DB wasn't found/ranked.
    # We treat "None" or "Rank > 1" as Ambiguous (1).
    if rank is None or rank > 1:
        label = 1
    else:
        label = 0

    features = {
        'concentration': top1_score,
        'margin': top1_score - top2_score,
        'entropy': entry.get('overall_entropy', 0), # Default to 0 if missing
        'is_ambiguous': label
    }
    extracted_data.append(features)

# 3. Create the DataFrame (The Table)
df = pd.DataFrame(extracted_data)

# 4. Define X (Features) and y (Target)
X = df[['concentration', 'margin', 'entropy']]
y = df['is_ambiguous']

# 5. Split, Train, and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Evaluate
predictions = model.predict(X_test)

print("--- Results on the Test Set ---")
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))