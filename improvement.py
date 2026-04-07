import json
from pathlib import Path

def analyze_router_performance(new_path, initial_path):
    # Load JSON data
    try:
        with open(new_path, 'r') as f:
            new_data = json.load(f)
        with open(initial_path, 'r') as f:
            initial_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        return

    # --- 1. EXTRACT RESULTS FROM NESTED STRUCTURE ---
    # Targets the "results" list inside results_train.json
    if isinstance(initial_data, dict) and "results" in initial_data:
        initial_list = initial_data["results"]
    else:
        initial_list = initial_data if isinstance(initial_data, list) else []

    # Create the lookup map (ID -> Object)
    initial_lookup = {str(item['question_id']): item for item in initial_list}

    # Normalize New Data (handles both list and dict-with-results-key)
    new_list = new_data if isinstance(new_data, list) else new_data.get("results", [])

    # --- 2. TRACKING ---
    stats = {
        "total": 0,
        "new_top1": 0,
        "init_top1": 0,
        "fixed": 0,      
        "regressed": 0,  
        "stayed_top1": 0
    }

    print(f"{'Q_ID':<10} | {'Init Rank':<10} | {'New Rank':<10} | {'Status'}")
    print("-" * 55)

    for new_item in new_list:
        qid = str(new_item.get('question_id'))
        
        if qid not in initial_lookup:
            continue
            
        init_item = initial_lookup[qid]
        stats["total"] += 1
        
        r_new = new_item.get('correct_db_rank')
        r_init = init_item.get('correct_db_rank')

        # Skip if ranks are missing
        if r_new is None or r_init is None:
            continue

        if r_new == 1: stats["new_top1"] += 1
        if r_init == 1: stats["init_top1"] += 1
        
        # Categorize the change
        if r_init > 1 and r_new == 1:
            stats["fixed"] += 1
            print(f"{qid:<10} | {r_init:<10} | {r_new:<10} | ⭐ FIXED")
        elif r_init == 1 and r_new > 1:
            stats["regressed"] += 1
            print(f"{qid:<10} | {r_init:<10} | {r_new:<10} | ❌ REGRESSION")
        elif r_init == 1 and r_new == 1:
            stats["stayed_top1"] += 1

    # --- 3. FINAL REPORT ---
    total = stats["total"]
    if total == 0:
        print("\nResult: No matching Question IDs found between the files.")
        return

    print("\n" + "="*45)
    print("         ROUTER IMPROVEMENT SUMMARY")
    print("="*45)
    print(f"Total Questions Matched:  {total}")
    print(f"Initial Top-1 Accuracy:   {(stats['init_top1']/total)*100:.2f}%")
    print(f"New Code Top-1 Accuracy:  {(stats['new_top1']/total)*100:.2f}%")
    print("-" * 45)
    print(f"Total Gains (New Wins):   {stats['fixed']}")
    print(f"Total Losses:             {stats['regressed']}")
    print(f"Maintained Successes:     {stats['stayed_top1']}")
    print("="*45)

# --- EXECUTION ---
# Ensure these paths match your local setup
base_dir = Path("/Users/anjalikamath/sem 6/SOP/Code")
new_file = base_dir / "llm_rerank_openrouter_2.json"
initial_file = base_dir / "results_train.json"

analyze_router_performance(new_file, initial_file)