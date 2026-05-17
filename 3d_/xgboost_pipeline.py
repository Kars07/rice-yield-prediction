import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def generate_xgboost_pipeline():
    print("Generating XGBoost Training Pipeline Diagram...")
    
    # Set up the figure (Vertical layout is best for Word documents)
    fig, ax = plt.subplots(figsize=(10, 11), dpi=300)
    ax.axis('off') # Hide axes
    
    box_width = 0.64
    x_start = 0.18 # Centers the 0.64 wide box on the 0.5 center line
    
    # ==========================================
    # 1. INPUT NODE
    # ==========================================
    y_input = 0.78
    h_input = 0.14
    box1 = mpatches.FancyBboxPatch((x_start, y_input), box_width, h_input, boxstyle="round,pad=0.02", 
                                   ec="black", fc="#f8fafc", lw=2)
    ax.add_patch(box1)
    
    # Header
    ax.text(0.5, y_input + h_input - 0.03, "Phase 1: Feature Ingestion (Tabular)", 
            ha='center', va='center', fontsize=13, fontweight='bold', color="black")
    ax.plot([x_start + 0.05, x_start + box_width - 0.05], [y_input + h_input - 0.06, y_input + h_input - 0.06], 
            color="black", lw=1, alpha=0.3)
    
    # Text
    text1 = (
        "• Extracted from the 70% Training Partition.\n"
        "• Features: NDVI_Mean, EVI_Max, NDWI_Mean,\n"
        "  VV_Mean, Total_Rain, Mean_Temp.\n"
        "• Normalized via StandardScaler (Z-score)."
    )
    # Perfectly centered in the lower area: y = 0.82
    ax.text(x_start + 0.05, 0.82, text1, ha='left', va='center', fontsize=11, linespacing=1.6, color="black")

    # Arrow 1
    ax.annotate('', xy=(0.5, y_input - 0.02), xytext=(0.5, y_input), 
                arrowprops=dict(arrowstyle="fancy,head_length=0.8,head_width=0.8,tail_width=0.3", color="black"))

    # ==========================================
    # 2. HYPERPARAMETER NODE
    # ==========================================
    y_hyper = 0.55
    h_hyper = 0.15
    box2 = mpatches.FancyBboxPatch((x_start, y_hyper), box_width, h_hyper, boxstyle="round,pad=0.02", 
                                   ec="black", fc="#fffbeb", lw=2)
    ax.add_patch(box2)
    
    ax.text(0.5, y_hyper + h_hyper - 0.03, "Phase 2: Hyperparameter Optimization", 
            ha='center', va='center', fontsize=13, fontweight='bold', color="black")
    ax.plot([x_start + 0.05, x_start + box_width - 0.05], [y_hyper + h_hyper - 0.06, y_hyper + h_hyper - 0.06], 
            color="black", lw=1, alpha=0.3)
    
    text2 = (
        "• Learning Rate (eta): 0.05 (Prevents overshooting)\n"
        "• Max Depth: 4 (Shallow trees to enforce generalization)\n"
        "• Subsample: 0.8 (Stochastic gradient boosting)\n"
        "• N_estimators: 100 Trees"
    )
    # Perfectly centered in the lower area: y = 0.595
    ax.text(x_start + 0.05, 0.595, text2, ha='left', va='center', fontsize=11, linespacing=1.6, color="black")

    # Arrow 2
    ax.annotate('', xy=(0.5, y_hyper - 0.02), xytext=(0.5, y_hyper), 
                arrowprops=dict(arrowstyle="fancy,head_length=0.8,head_width=0.8,tail_width=0.3", color="black"))

    # ==========================================
    # 3. TRAINING NODE
    # ==========================================
    y_train = 0.28
    h_train = 0.19
    box3 = mpatches.FancyBboxPatch((x_start, y_train), box_width, h_train, boxstyle="round,pad=0.02", 
                                   ec="black", fc="#f0fdf4", lw=2)
    ax.add_patch(box3)
    
    ax.text(0.5, y_train + h_train - 0.03, "Phase 3: Model Training with Early Stopping", 
            ha='center', va='center', fontsize=13, fontweight='bold', color="black")
    ax.plot([x_start + 0.05, x_start + box_width - 0.05], [y_train + h_train - 0.06, y_train + h_train - 0.06], 
            color="black", lw=1, alpha=0.3)
    
    text3 = (
        "• Sequential Tree Construction: Each new decision tree\n"
        "  corrects the residual errors of the previous ensemble.\n"
        "• Objective Function: Mean Squared Error (MSELoss).\n"
        "• Early Stopping Protocol: Training halts immediately if\n"
        "  the Validation Error stagnates for 10 iterations, strictly\n"
        "  preventing data overfitting."
    )
    # Perfectly centered in the lower area: y = 0.345
    ax.text(x_start + 0.05, 0.345, text3, ha='left', va='center', fontsize=11, linespacing=1.6, color="black")

    # Arrow 3
    ax.annotate('', xy=(0.5, y_train - 0.02), xytext=(0.5, y_train), 
                arrowprops=dict(arrowstyle="fancy,head_length=0.8,head_width=0.8,tail_width=0.3", color="black"))

    # ==========================================
    # 4. OUTPUT NODE
    # ==========================================
    y_out = 0.06
    h_out = 0.14
    box4 = mpatches.FancyBboxPatch((x_start, y_out), box_width, h_out, boxstyle="round,pad=0.02", 
                                   ec="black", fc="#eff6ff", lw=2)
    ax.add_patch(box4)
    
    ax.text(0.5, y_out + h_out - 0.03, "Phase 4: Output & Serialization", 
            ha='center', va='center', fontsize=13, fontweight='bold', color="black")
    ax.plot([x_start + 0.05, x_start + box_width - 0.05], [y_out + h_out - 0.06, y_out + h_out - 0.06], 
            color="black", lw=1, alpha=0.3)
    
    text4 = (
        "• Trained XGBoost Regressor instance generated.\n"
        "• Evaluated on Test Set (15%) for final RMSE.\n"
        "• Serialized and exported as xgboost_model_v2.json\n"
        "• Ready for Ensemble Fusion and API deployment."
    )
    # Perfectly centered in the lower area: y = 0.10
    ax.text(x_start + 0.05, 0.10, text4, ha='left', va='center', fontsize=11, linespacing=1.6, color="black")

    # ==========================================
    # TITLE
    # ==========================================
    ax.set_title("Figure 4.11: XGBoost Training and Optimization Pipeline", fontsize=15, fontweight='bold', y=0.96)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Save the plot
    plt.savefig('Figure_4_11_XGBoost_Pipeline.png', bbox_inches='tight')
    print("-> Successfully saved 'Figure_4_11_XGBoost_Pipeline.png'")

if __name__ == "__main__":
    generate_xgboost_pipeline()