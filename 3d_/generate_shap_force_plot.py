import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def generate_shap_force_plot():
    print("Generating Figure 4.22: SHAP Additive Force Plot...")
    
    # 1. Initialize Canvas Frame Layout
    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=300)
    ax.axis('off')
    ax.set_xlim(1.5, 3.8)
    ax.set_ylim(0, 10)
    
    # Mathematical Reference Scale Definitions (t/ha)
    base_val = 2.15
    final_val = 3.12
    
    # 2. Draw Horizontal Yield Metric Scale Axis Line
    ax.plot([1.6, 3.7], [2, 2], color='black', linewidth=1.5, zorder=1)
    
    # Structural Tick Marks on Axis Timeline
    ticks = [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6]
    for t in ticks:
        ax.plot([t, t], [1.8, 2.2], color='black', linewidth=1.2, zorder=2)
        ax.text(t, 1.4, f"{t:.2f}", ha='center', va='top', fontsize=10, fontweight='bold', color='#334155')
        
    ax.text(2.65, 0.6, "Prediction Value Scale: Rice Yield (t/ha)", ha='center', fontsize=11, fontweight='bold', color='black')

    # ==========================================
    # 3. BASELINE AND OUTPUT ANNOTATION ANCHORS
    # ==========================================
    # Base Value Vector Pointer (Expected Dataset Average)
    ax.annotate('', xy=(base_val, 2.0), xytext=(base_val, 3.2),
                arrowprops=dict(arrowstyle="->", color='#475569', lw=2))
    ax.text(base_val, 3.4, f"Base Value\nE[f(X)] = {base_val:.2f} t/ha", ha='center', va='bottom',
            fontsize=10.5, fontweight='bold', color='#475569',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#cbd5e1', boxstyle='round,pad=0.3'))

    # Final Output Prediction Vector Pointer (Actual System Calculation)
    ax.annotate('', xy=(final_val, 2.0), xytext=(final_val, 3.2),
                arrowprops=dict(arrowstyle="->", color='#1e3a8a', lw=2.5))
    ax.text(final_val, 3.4, f"Model Output\nf(X) = {final_val:.2f} t/ha", ha='center', va='bottom',
            fontsize=11.5, fontweight='bold', color='#1e3a8a',
            bbox=dict(facecolor='#eff6ff', alpha=0.9, edgecolor='#1d4ed8', boxstyle='round,pad=0.4'))

    # ==========================================
    # 4. QUANTIFY ADDITIVE ATTRIBUTION FORCES
    # ==========================================
    y_blocks = 5.2
    h_blocks = 1.0
    
    # --- POSITIVE ACCELERATORS (Crimson Shades Pushing Right) ---
    # Driver 1: High Canopy Biomass (EVI_Max)
    rect1 = mpatches.Rectangle((2.15, y_blocks), 0.65, h_blocks, facecolor='#fca5a5', edgecolor='#dc2626', lw=1.5, zorder=3)
    ax.add_patch(rect1)
    ax.text(2.475, y_blocks + h_blocks/2, "EVI_Max = 0.52\n(+0.65)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#7f1d1d')
    
    # Driver 2: Ample Rainfall Accumulations (Total_Rain)
    rect2 = mpatches.Rectangle((2.80, y_blocks), 0.42, h_blocks, facecolor='#fecaca', edgecolor='#ef4444', lw=1.5, zorder=3)
    ax.add_patch(rect2)
    ax.text(3.01, y_blocks + h_blocks/2, "Total_Rain\n(+0.42)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#7f1d1d')

    # Driver 3: Healthy Moisture Thickness (NDWI_Mean)
    rect3 = mpatches.Rectangle((3.22, y_blocks), 0.18, h_blocks, facecolor='#fee2e2', edgecolor='#f87171', lw=1.5, zorder=3)
    ax.add_patch(rect3)
    ax.text(3.31, y_blocks + h_blocks/2, "NDWI\n(+0.18)", ha='center', va='center', fontsize=8, fontweight='bold', color='#7f1d1d')

    # --- NEGATIVE BRAKING ELEMENTS (Blue Shades Pulling Left) ---
    # Drag 1: Thermal Stress Waves (Mean_Temp)
    rect4 = mpatches.Rectangle((3.12, y_blocks), 0.28, h_blocks, facecolor='#bfdbfe', edgecolor='#2563eb', lw=1.5, alpha=0.5, zorder=2)
    # Highlight block outline specifically on the left face where force vectors pull back
    ax.plot([3.40, 3.12], [y_blocks + h_blocks, y_blocks + h_blocks], color='#2563eb', linewidth=1.5, zorder=4)
    ax.text(3.31, y_blocks - 1.2, "Mean_Temp = 34.2°C\n(-0.18)", ha='center', va='center', fontsize=9, fontweight='bold', color='#1e3a8a')
    ax.plot([3.31, 3.31], [y_blocks, y_blocks - 0.5], color='#2563eb', linestyle=':', lw=1)

    # Drag 2: Rough Surface Backscattering Fluctuations (VV_SAR_Mean)
    ax.text(3.06, y_blocks - 2.4, "VV_SAR_Mean\n(-0.10)", ha='center', va='center', fontsize=8, fontweight='bold', color='#1e3a8a')
    ax.plot([3.06, 3.17], [y_blocks, y_blocks - 1.8], color='#3b82f6', linestyle=':', lw=1)

    # ==========================================
    # 5. DOCUMENT MASTER TITLES & METRIC OVERLAYS
    # ==========================================
    ax.text(2.65, 9.5, "Figure 4.22: SHAP Force Plot Instance for an Individual Growing Season", ha='center', va='top', fontsize=14, fontweight='bold', color='black')
    ax.text(3.29, 9.0, "Visualizing Feature-Level Additive Contributions Shifting the Baselined National Prediction Matrix", ha='center', va='top', fontsize=11, style='italic', color='#475569')

    # Structural Monospace Context Panel Block positioned safely on the far left margin
    props = dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='black', lw=1.0)
    context_text = (
        "Instance Interpretation Profile:\n"
        "───────────────────────────────\n"
        "• Target Specimen: Taraba State Partition, 2023 Growing Cycle.\n"
        "• Baseline Context: Model unconditioned expected value sits at 2.15 t/ha.\n"
        "• Dominant Catalyst: EVI_Max (+0.65 t/ha) drives major yield expansion.\n"
        "• Countervailing Drag: High heat stress peaks introduce a -0.18 t/ha penalty."
    )
    ax.text(1.6, 9.1, context_text, transform=ax.transData, fontsize=9.5, fontweight='bold',
            va='top', ha='left', bbox=props, fontfamily='monospace', linespacing=1.4)

    plt.tight_layout()
    plt.savefig('Figure_4_22_SHAP_Force_Plot.png', bbox_inches='tight')
    print("-> Successfully saved clean 'Figure_4_22_SHAP_Force_Plot.png'")

if __name__ == "__main__":
    generate_shap_force_plot()