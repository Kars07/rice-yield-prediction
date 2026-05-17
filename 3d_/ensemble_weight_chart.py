import numpy as np
import matplotlib.pyplot as plt

def generate_ensemble_weight_chart():
    print("Generating Academic Ensemble Weight Contribution Chart...")
    
    # 1. Establish validated model parameters
    weights = [51.71, 48.29]
    labels = ['XGBoost Regressor\n(Tabular Surface Dynamics)', 'Stacked LSTM Network\n(Deep Temporal Phenology)']
    colors = ['#1e293b', '#e2e8f0'] # Academic High-Contrast Monochrome Palette
    
    # 2. Initialize figure structure
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    
    # 3. Create horizontal bars with crisp black borders
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, weights, color=colors, edgecolor='black', height=0.55, linewidth=1.2)
    
    # 4. Inject explicit data percentage labels inside/beside the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        # Choose text color based on bar background contrast
        text_color = 'white' if i == 0 else 'black'
        
        ax.text(width / 2 if i == 0 else width - 6, 
                bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}%', 
                va='center', ha='center', 
                fontsize=11, fontweight='bold', color=text_color)
                
    # 5. Accentuate axis layouts and typography
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, fontweight='bold', color='black')
    ax.set_xlabel('Meta-Weight Contribution Probability (%)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Figure 4.18: Empirical Weight Distribution of the Hybrid Ensemble Architecture', 
                 fontsize=12, fontweight='bold', pad=18)
    
    # 6. Enforce hard grid boundaries up to 100%
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True, linestyle=':', alpha=0.4, color='gray')
    ax.set_axisbelow(True) # Ensure grid sits strictly behind the bars
    
    # 7. Format clean structural borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # 8. Add an analytical methodology description block at the bottom
    justification_text = (
        "Optimization Protocol: Meta-weights were derived via historical variance minimization on the validation \n"
        "partition. The 51.71% priority allocation to XGBoost captures abrupt localized environmental and climatic shocks, \n"
        "while the 48.29% allocation to the LSTM ensures continuous sequence alignment across the crop's vegetative cycles."
    )
    fig.text(0.1, -0.12, justification_text, ha='left', va='top', fontsize=9.5, style='italic', color='#334155', linespacing=1.4)

    plt.tight_layout()
    plt.savefig('Figure_4_18_Ensemble_Weights.png', bbox_inches='tight')
    print("-> Successfully saved 'Figure_4_18_Ensemble_Weights.png'")

if __name__ == "__main__":
    generate_ensemble_weight_chart()