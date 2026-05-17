import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.patches as mpatches

def generate_3d_block_ensemble_workflow():
    print("Generating Spaced 3D Solid Block Ensemble Diagram...")
    
    fig = plt.figure(figsize=(16, 10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off() 

    # Helper function to draw a clean, solid 3D block with strict borders
    def draw_solid_block(x_start, y_start, z_start, dx, dy, dz, face_color, edge_color='black'):
        # Define vertices for a solid 3D box
        vertices = np.array([
            [x_start, y_start, z_start],
            [x_start + dx, y_start, z_start],
            [x_start + dx, y_start + dy, z_start],
            [x_start, y_start + dy, z_start],
            [x_start, y_start, z_start + dz],
            [x_start + dx, y_start, z_start + dz],
            [x_start + dx, y_start + dy, z_start + dz],
            [x_start, y_start + dy, z_start + dz]
        ])
        
        # Define the 6 faces
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]], # Bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]], # Top
            [vertices[0], vertices[1], vertices[5], vertices[4]], # Front
            [vertices[2], vertices[3], vertices[7], vertices[6]], # Back
            [vertices[0], vertices[3], vertices[7], vertices[4]], # Left
            [vertices[1], vertices[2], vertices[6], vertices[5]]  # Right
        ]
        
        poly = Poly3DCollection(faces, facecolors=face_color, edgecolors=edge_color, linewidths=1.2, alpha=1.0)
        ax.add_collection3d(poly)

    # ==========================================
    # 1. DRAW PARALLEL PROCESSING BLOCKS (SPACED)
    # ==========================================
    
    # --- Top Engine: XGBoost ---
    # Moved to Y=6 to create a massive central gap
    draw_solid_block(x_start=0, y_start=4.5, z_start=2, dx=2.5, dy=3.0, dz=2.0, face_color='#fffbeb')
    ax.text(1.25, 6.0, 5.0, "XGBoost Engine", ha='center', va='center', fontsize=11, fontweight='bold', zorder=100)
    ax.text(1.25, 6.0, 1.0, "Input: Tabular\n[Batch, 7]", ha='center', va='center', fontsize=9, style='italic', zorder=100)

    # --- Bottom Engine: LSTM ---
    # Moved down to Y=-7.5 to mirror the spacing completely
    draw_solid_block(x_start=0, y_start=-7.5, z_start=2, dx=2.5, dy=3.0, dz=2.0, face_color='#f0fdf4')
    ax.text(1.25, -6.0, 5.0, "LSTM Network", ha='center', va='center', fontsize=11, fontweight='bold', zorder=100)
    ax.text(1.25, -6.0, 1.0, "Input: Sequences\n[Batch, 10, 7]", ha='center', va='center', fontsize=9, style='italic', zorder=100)

    # ==========================================
    # 2. DRAW THE AGGREGATION NODE (CENTERED)
    # ==========================================
    # Perfectly centered on the Y-axis (Y=-1.5 to 1.5) and shifted right down the pipeline (X=9)
    draw_solid_block(x_start=9, y_start=-2.0, z_start=-0.5, dx=3.5, dy=4.0, dz=2.5, face_color='#eff6ff')
    ax.text(10.75, 0, 3.0, "Hybrid Fusion Hub", ha='center', va='center', fontsize=12, fontweight='bold', zorder=100)
    
    # Weights placed safely clear of the box structure below it
    ax.text(10.75, 0, -2.0, "w1 (XGBoost): 51.71%\nw2 (LSTM): 48.29%", ha='center', va='center', 
            fontsize=10, fontweight='600', color='black', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'), zorder=100)

    # ==========================================
    # 3. DRAW THE PREDICTION OUTPUT (FINAL)
    # ==========================================
    # Shifting out further to the right (X=18)
    draw_solid_block(x_start=18, y_start=-1.5, z_start=-3.0, dx=2.0, dy=3.0, dz=1.5, face_color='#fef2f2')
    ax.text(19.0, 0, -0.5, "Integrated Output", ha='center', va='center', fontsize=11, fontweight='bold', zorder=100)
    ax.text(19.0, 0, -4.5, "Rice Yield\n(Vector: t/ha)", ha='center', va='center', fontsize=10, fontweight='bold', color='#991b1b', zorder=100)

    # ==========================================
    # 4. CRISP PIPELINE VECTOR LINES
    # ==========================================
    # Solid clean trajectories with no intersecting text paths
    ax.plot([2.5, 9], [6.0, 1.0], [3.0, 0.75], color='black', linewidth=2, linestyle='-')
    ax.plot([2.5, 9], [-6.0, -1.0], [3.0, 0.75], color='black', linewidth=2, linestyle='-')
    ax.plot([12.5, 18], [0, 0], [0.75, -2.25], color='black', linewidth=2, linestyle='-')

    # ==========================================
    # 5. FORMAL TEXT INFORMATION OVERLAYS
    # ==========================================
    fig.text(0.5, 0.93, "Figure 4.14: Hybrid Ensemble Training and Fusion Workflow", 
             ha='center', fontsize=18, fontweight='bold', color='black')
    fig.text(0.5, 0.90, "Parallel Non-Linear Gradient Boosting and Deep Temporal Aggregation Pipeline", 
             ha='center', fontsize=12, color='#333333')

    # Mathematical Formula Definition block kept strictly flat 2D on the left margin
    props = dict(boxstyle='round,pad=0.6', facecolor='#f8fafc', edgecolor='black', lw=1.5)
    formula_text = (
        "MATHEMATICAL FUSION EQUATION:\n"
        "----------------------------------------------------------\n"
        "Ŷ_final = (w1 × Ŷ_XGBoost) + (w2 × Ŷ_LSTM)\n\n"
        "Where Optimized Empirical Metaweights are:\n"
        "• w1 = 0.5171 (XGBoost Regressor Priority Weight)\n"
        "• w2 = 0.4829 (LSTM Temporal Memory Weight)\n\n"
        "Rationale: Weighted variance minimization across models\n"
        "yields a structurally superior boundary generalization."
    )
    fig.text(0.12, 0.12, formula_text, fontsize=11, fontweight='bold', 
             color='#0f172a', bbox=props, linespacing=1.6)

    # ==========================================
    # 6. GRAPH LIMITS & CAMERA CONTROL
    # ==========================================
    # High-elevation side look to maintain distinct depth profiles
    ax.view_init(elev=22, azim=-55)
    
    ax.set_xlim([-2, 22])
    ax.set_ylim([-9, 9])
    ax.set_zlim([-6, 7])
    
    plt.savefig('Figure_4_14_Ensemble_Workflow.png', bbox_inches='tight')
    print("-> Successfully saved 'Figure_4_14_Ensemble_Workflow.png'")

if __name__ == "__main__":
    generate_3d_block_ensemble_workflow()