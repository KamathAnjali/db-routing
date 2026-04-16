import json
import matplotlib.pyplot as plt

def plot_top1_top2_margin_distribution(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    margins = []
    
    for entry in data:
        top10 = entry.get("top10", [])
        if len(top10) >= 2:
            margin = top10[0]["llm_score"] - top10[1]["llm_score"]
            margins.append(margin)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(margins, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_xlabel('Score Margin (Top 1 - Top 2)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('LLM Score Margin Distribution — Top 1 vs Top 2', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('top1_top2_margin_distribution.png', dpi=150, bbox_inches='tight')
    print("Saved to top1_top2_margin_distribution.png")

plot_top1_top2_margin_distribution("rank_1.json")