import json
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_top1_llm_score_distribution(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    all_scores = []
    
    for entry in data:
        top10 = entry.get("top10", [])
        if top10:
            all_scores.append(top10[0]["llm_score"])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_scores, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_xlabel('LLM Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('LLM Score Distribution — Top 1 Database per Query', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('top1_llm_score_distribution.png', dpi=150, bbox_inches='tight')
    print("Saved to top1_llm_score_distribution.png")

plot_top1_llm_score_distribution("rank_1.json")