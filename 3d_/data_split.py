import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def generate_flawless_academic_split():
    print("Generating Flawless Academic Data Split Diagram...")
    
    fig, ax = plt.subplots(figsize=(14, 8.5), dpi=300) 
    ax.axis('off') # Hide axes
    
    # --- 1. MAIN DATASET BOX (Top) ---
    full_data = mpatches.FancyBboxPatch((0.1, 0.83), 0.8, 0.10, boxstyle="round,pad=0.02", 
                                        ec="black", fc="white", lw=2)
    ax.add_patch(full_data)
    ax.text(0.5, 0.90, "Processed National Multi-Sensor Dataset ( national_processed_v2.csv )", 
            ha='center', va='center', fontsize=13, fontweight='bold', color="black")
    ax.text(0.5, 0.86, "Temporal Range: May 2022 – December 2025 | Aggregation: 10-Step Sequential Phenology", 
            ha='center', va='center', fontsize=11, color="black")
            
    # Connecting Arrows Down
    arrow_props = dict(arrowstyle="->", lw=2, color="black")
    ax.annotate('', xy=(0.24, 0.77), xytext=(0.24, 0.83), arrowprops=arrow_props)
    ax.annotate('', xy=(0.575, 0.77), xytext=(0.575, 0.83), arrowprops=arrow_props)
    ax.annotate('', xy=(0.835, 0.77), xytext=(0.835, 0.83), arrowprops=arrow_props)

    # --- 2. TRAINING SET (Left) ---
    train_box = mpatches.FancyBboxPatch((0.05, 0.30), 0.38, 0.47, boxstyle="round,pad=0.02", 
                                        ec="black", fc="white", lw=2)
    ax.add_patch(train_box)
    ax.text(0.24, 0.74, "Model Training Set (70%)", ha='center', fontsize=12, fontweight='bold', color="black")
    ax.text(0.24, 0.70, "Timeframe: 2022 to Mid-2024", ha='center', fontsize=10, fontweight='bold', color="#333333")
    ax.plot([0.07, 0.41], [0.67, 0.67], color="black", lw=1, alpha=0.5)
    
    train_details = (
        "• Largest temporal partition.\n\n"
        "• Used to optimize XGBoost\n  decision trees.\n\n"
        "• Used to train LSTM sequence\n  memory weights.\n\n"
        "• Establishes baseline curves."
    )
    ax.text(0.07, 0.64, train_details, ha='left', va='top', fontsize=10.5, color="black", linespacing=1.6)

    # --- 3. VALIDATION SET (Middle) ---
    val_box = mpatches.FancyBboxPatch((0.45, 0.30), 0.25, 0.47, boxstyle="round,pad=0.02", 
                                      ec="black", fc="white", lw=2)
    ax.add_patch(val_box)
    ax.text(0.575, 0.74, "Validation Set (15%)", ha='center', fontsize=12, fontweight='bold', color="black")
    ax.text(0.575, 0.70, "Timeframe: Late 2024", ha='center', fontsize=10, fontweight='bold', color="#333333")
    ax.plot([0.47, 0.68], [0.67, 0.67], color="black", lw=1, alpha=0.5)

    val_details = (
        "• Hyperparameter tuning.\n\n"
        "• Triggers Early Stopping.\n\n"
        "• Calculates Ensemble Weights."
    )
    ax.text(0.47, 0.64, val_details, ha='left', va='top', fontsize=10.5, color="black", linespacing=1.6)

    # --- 4. TESTING SET (Right) ---
    test_box = mpatches.FancyBboxPatch((0.72, 0.30), 0.23, 0.47, boxstyle="round,pad=0.02", 
                                       ec="black", fc="white", lw=2)
    ax.add_patch(test_box)
    ax.text(0.835, 0.74, "Testing Set (15%)", ha='center', fontsize=12, fontweight='bold', color="black")
    ax.text(0.835, 0.70, "Timeframe: 2025", ha='center', fontsize=10, fontweight='bold', color="#333333")
    ax.plot([0.74, 0.93], [0.67, 0.67], color="black", lw=1, alpha=0.5)

    test_details = (
        "• Strictly Unseen Data.\n\n"
        "• Simulates real-world\n  future forecasting.\n\n"
        "• Final RMSE evaluation."
    )
    ax.text(0.74, 0.64, test_details, ha='left', va='top', fontsize=10.5, color="black", linespacing=1.6)

    # --- 5. THE TIMELINE ARROW (Bottom) ---
    ax.annotate('', xy=(0.95, 0.22), xytext=(0.05, 0.22), 
                arrowprops=dict(arrowstyle="fancy,head_length=0.8,head_width=0.8,tail_width=0.3", lw=1, color='black', fc='black'))
    ax.text(0.5, 0.18, "Chronological Arrow of Time (Sequence Preservation)", ha='center', fontsize=11, fontweight='bold', color='black')

    # --- 6. CRITICAL JUSTIFICATION BOX (Bottom) ---
    # WIDENED BOX: Starts at 0.08, width is 0.84 (Spans much wider to easily hold the text)
 # INCREASED WIDTH (0.90) AND HEIGHT (0.15), shifted left (0.05) to stay centered
    warning_box = mpatches.FancyBboxPatch((0.05, 0.01), 0.90, 0.13, boxstyle="round,pad=0.02", 
                                          ec="black", fc="#f8f9fa", lw=2, linestyle='--')
    ax.add_patch(warning_box)
    
    # REFORMATTED TEXT: Added an extra line break to ensure it forms a clean, compact block
    warning_text = (
        "ARCHITECTURAL JUSTIFICATION: STRICT TEMPORAL SPLITTING\n"
        "Random cross-validation was strictly prohibited. The data was split sequentially to prevent 'Temporal Data Leakage'.\n"
        "This ensures the LSTM network learns from historical patterns to predict future harvests,\n"
        "mimicking real-world deployment."
    )
    ax.text(0.5, 0.08, warning_text, ha='center', va='center', fontsize=10, fontweight='bold', color="black", linespacing=1.5)

    # --- TITLE ---
    ax.set_title("Time-Series Aware Data Splitting Strategy", fontsize=14, fontweight='bold', y=0.98)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Save the plot
    plt.savefig('Figure_4_10_Flawless_Data_Split.png', bbox_inches='tight')
    print("-> Successfully saved 'Figure_4_10_Flawless_Data_Split.png'")

if __name__ == "__main__":
    generate_flawless_academic_split()