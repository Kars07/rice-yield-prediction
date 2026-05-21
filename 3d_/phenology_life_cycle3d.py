import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm

def generate_bold_combined_3d_space():
    print("Generating Figure 4.26: Bold 3D Parametric & Histographic Space...")
    
    # 1. Initialize High-Resolution 3D Canvas
    fig = plt.figure(figsize=(16, 13), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate continuous timeline for the trajectory curve
    t_curve = np.linspace(0, 10, 600)
    x_curve = t_curve * 21  # 0 to 210 growing days
    y_curve = 0.15 + 0.68 * np.exp(-0.5 * ((t_curve - 5.8) / 1.9) ** 2)  # EVI curve
    z_curve = 1.0 / (1.0 + np.exp(-0.8 * (t_curve - 4.5))) * 8.5 + 0.5   # Memory Activation
    
    # 2. Render the 3D Histographic Bars (Minimalist Pillars)
    num_bins = 10
    t_bins = np.linspace(0.5, 9.5, num_bins)
    x_bars = t_bins * 21
    y_bars = 0.15 + 0.68 * np.exp(-0.5 * ((t_bins - 5.8) / 1.9) ** 2)
    
    bar_width = 3.0   
    bar_depth = 0.015 
    bar_cmap = plt.get_cmap('GnBu')
    
    for idx in range(num_bins):
        x_pos = x_bars[idx] - bar_width / 2.0
        y_pos = y_bars[idx] - bar_depth / 2.0
        z_pos = 0  
        
        height = 1.0 / (1.0 + np.exp(-0.8 * (t_bins[idx] - 4.5))) * 8.5 + 0.5
        color_val = bar_cmap(0.15 + (height / 10.0) * 0.5)
        
        # Keep bars translucent and borderless so they don't clog the view
        ax.bar3d(x_pos, y_pos, z_pos, bar_width, bar_depth, height, 
                 color=color_val, edgecolor='none', alpha=0.35, zorder=2)

    # 3. Render the Continuous Parametric 3D Trajectory Curve (BOLD & FAT)
    line_cmap = plt.get_cmap('YlGnBu')
    for i in range(len(t_curve) - 1):
        color_ratio = i / len(t_curve)
        # FIX: Increased linewidth to 7.5, max alpha, and added round capstyles for a smooth tube effect
        ax.plot(x_curve[i:i+2], y_curve[i:i+2], z_curve[i:i+2], 
                color=line_cmap(0.3 + color_ratio * 0.7), linewidth=7.5, alpha=1.0, 
                solid_capstyle='round', zorder=15)

    # 4. Inject Segmented 3D Scattered Phase Nodes (Proportionally scaled up)
    stages = [
        {"name": "1. Seedling & Transplanting", "mask": (t_curve >= 0) & (t_curve < 2.5), "color": "#22c55e", "marker": "o", "size": 80},
        {"name": "2. Vegetative & Tillering", "mask": (t_curve >= 2.5) & (t_curve < 5.5), "color": "#15803d", "marker": "^", "size": 110},
        {"name": "3. Reproductive & Heading", "mask": (t_curve >= 5.5) & (t_curve < 8.0), "color": "#eab308", "marker": "s", "size": 110},
        {"name": "4. Ripening & Senescence", "mask": (t_curve >= 8.0) & (t_curve <= 10.0), "color": "#b45309", "marker": "D", "size": 80}
    ]
    
    for stage in stages:
        mask = stage["mask"]
        ax.scatter(x_curve[mask][::10], y_curve[mask][::10], z_curve[mask][::10], 
                   color=stage["color"], marker=stage["marker"], s=stage["size"], 
                   edgecolor='black', linewidth=0.8, alpha=1.0, label=stage["name"], zorder=20)

    # 5. Render Floor Drop-Shadow Path Projection
    # Make the shadow slightly bolder as well to match the main curve
    ax.plot(x_curve, y_curve, zs=0, zdir='z', color='#94a3b8', linestyle='--', linewidth=2.5, alpha=0.6, zorder=1)
    
    x_poly = np.concatenate([x_curve, [x_curve[-1], x_curve[0]]])
    y_poly = np.concatenate([y_curve, [0, 0]])
    z_poly = np.zeros_like(x_poly)
    verts = [list(zip(x_poly, y_poly, z_poly))]
    
    shadow_poly = Poly3DCollection(verts, facecolors='#cbd5e1', edgecolors='none', alpha=0.15, zorder=0)
    ax.add_collection3d(shadow_poly)

    # =========================================================================
    # 6. GRAPHICAL FORMATTING AND BLUEPRINT GRID LINES
    # =========================================================================
    ax.set_xlim(0, 210)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0, 10)
    
    ax.set_xlabel("Temporal Timeline Stride (Days from Inundation)", fontsize=11, fontweight='bold', labelpad=15)
    ax.set_ylabel("Satellite Canopy Reflection (EVI Scale Index)", fontsize=11, fontweight='bold', labelpad=15)
    ax.set_zlabel("LSTM Latent State Integration Intensity (Memory)", fontsize=11, fontweight='bold', labelpad=15)
    
    ax.set_title("Unified 3D Parametric Trajectory and Histographic Phenology Space", 
                 fontsize=14, fontweight='bold', pad=25)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#e2e8f0')
    ax.yaxis.pane.set_edgecolor('#e2e8f0')
    ax.zaxis.pane.set_edgecolor('#e2e8f0')
    ax.grid(True, linestyle=':', alpha=0.3)

    ax.view_init(elev=18, azim=-66)
    
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.88), title="Biological Crop Progressions", 
              title_fontproperties={'weight': 'bold'}, frameon=True, edgecolor='black', facecolor='#f8fafc', fontsize=10)

    explanation_text = (
        "Combined Spatiotemporal Architecture: This hybrid visualizer combines continuous trajectories with discrete 3D histograms.\n"
        "The continuous gradient line maps the biological transition paths over a 210-day cycle. Concurrently, the 3D bar blocks capture the discrete\n"
        "activation volume inside the 10 sequential lookback windows, explicitly showing how memory weight compounds toward maturity."
    )
    fig.text(0.5, 0.04, explanation_text, ha='center', va='top', fontsize=11, style='italic', color='#1e293b', fontweight='bold')

    plt.tight_layout()
    plt.savefig('Figure_4_26_Combined_3D_Phenology_Space_Bold.png', bbox_inches='tight')
    print("-> Successfully saved pristine, bold 'Figure_4_26_Combined_3D_Phenology_Space_Bold.png'")

if __name__ == "__main__":
    generate_bold_combined_3d_space()