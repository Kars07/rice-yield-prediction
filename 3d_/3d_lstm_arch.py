import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.patches as mpatches

def generate_3d_glass_pane_architecture():
    print("Generating 3D Holographic Glass-Pane Architecture Diagram...")
    
    fig = plt.figure(figsize=(16, 9), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off() # Hide axes for clean floating aesthetic
    
    # ==========================================
    # 1. DEFINE THE TENSOR PANES (LAYERS)
    # ==========================================
    # Format: (X_pos, Width(Y), Height(Z), FaceColor, EdgeColor, Title, TensorShape)
    # The physical area (Width x Height) visually scales with the number of neurons
    layers = [
        (0,  7.0, 10.0, '#bfdbfe', '#1e3a8a', 'Input Sequence', '[Batch, 10, 7]'),
        (7,  8.0,  8.0, '#bbf7d0', '#14532d', 'LSTM Layer 1\n(64 Units)', '[Batch, 10, 64]'),
        (14, 8.0,  8.0, '#4ade80', '#16a34a', 'LSTM Layer 2\n(64 Units)', '[Batch, 64]*'),
        (21, 4.0,  4.0, '#fef08a', '#ca8a04', 'Dense (FC) Layer\n(16 Units)', '[Batch, 16]'),
        (27, 1.0,  1.0, '#fca5a5', '#dc2626', 'Final Output\n(Yield t/ha)', '[Batch, 1]')
    ]

    pane_corners = []

    # Function to draw a glowing 3D glass pane
    def draw_glass_pane(x, w, h, f_color, e_color):
        # Calculate the 4 corners of the pane centered on the X-axis
        y_min, y_max = -w/2, w/2
        z_min, z_max = -h/2, h/2
        
        # Vertices of the rectangle
        verts = [[(x, y_min, z_min), (x, y_max, z_min), 
                  (x, y_max, z_max), (x, y_min, z_max)]]
        
        # Create the 3D polygon (High transparency for "glass" effect)
        poly = Poly3DCollection(verts, facecolors=f_color, alpha=0.35, 
                                edgecolors=e_color, linewidths=2.5)
        ax.add_collection3d(poly)
        
        # Return corners for drawing connecting frustums
        return [(x, y_min, z_min), (x, y_max, z_min), (x, y_max, z_max), (x, y_min, z_max)]

    # ==========================================
    # 2. RENDER PANES AND TEXT LABELS
    # ==========================================
    for i, (x, w, h, f_color, e_color, title, shape) in enumerate(layers):
        corners = draw_glass_pane(x, w, h, f_color, e_color)
        pane_corners.append(corners)
        
        # Floating Title above the pane
        ax.text(x, 0, 7, title, ha='center', va='center', zdir='y',
                fontsize=11, fontweight='bold', color='black', 
                bbox=dict(facecolor='white', alpha=0.85, edgecolor=e_color, boxstyle='round,pad=0.3'))
                
        # PyTorch Tensor Shape floating below the pane
        ax.text(x, 0, -6.5, shape, ha='center', va='center', zdir='y',
                fontsize=10, fontweight='bold', color='#334155', style='italic')

    # ==========================================
    # 3. DRAW CONNECTING FRUSTUMS (Data Compression)
    # ==========================================
    # Draw dashed lines connecting the corners to show how the tensors mathematically compress
    for i in range(len(pane_corners) - 1):
        curr = pane_corners[i]
        nxt = pane_corners[i+1]
        
        for j in range(4): # Connect the 4 corners
            ax.plot([curr[j][0], nxt[j][0]], [curr[j][1], nxt[j][1]], [curr[j][2], nxt[j][2]], 
                    color='#94a3b8', linestyle='--', linewidth=1.5, alpha=0.6)
            
        # Draw a central "Data Flow" arrow passing through the middle of the panes
        ax.plot([layers[i][0], layers[i+1][0]], [0, 0], [0, 0], 
                color=layers[i+1][4], linestyle='-', linewidth=2, alpha=0.8)

    # ==========================================
    # 4. 2D OVERLAYS AND ANNOTATIONS
    # ==========================================
    # Titles
    fig.text(0.5, 0.92, "Figure 4.13: LSTM Holographic Network Architecture", 
             ha='center', fontsize=18, fontweight='bold', color='black')
    fig.text(0.5, 0.89, "Visualizing Tensor Dimensionality Reduction from Input to Prediction", 
             ha='center', fontsize=12, color='#333333')

    # The Architecture Note Box (Bottom Left)
    props = dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='black', lw=1.5)
    arch_text = (
        "PYTORCH TENSOR TRANSFORMATIONS:\n"
        "--------------------------------------------------\n"
        "1. Input: (Batch, 10 Timesteps, 7 Features)\n"
        "2. LSTM 1: Extracts temporal patterns (64 units)\n"
        "3. LSTM 2: Deepens pattern recognition (64 units)\n"
        "4. *Sequence Extraction: lstm_out[:, -1, :]\n"
        "   (Drops the sequence dimension to output a 1D vector)\n"
        "5. Dense: Dropout (0.2) applied, compresses to 16 units\n"
        "6. Output: Final Linear layer outputs single Yield value"
    )
    fig.text(0.12, 0.12, arch_text, fontsize=10.5, fontweight='bold', 
             color='#0f172a', bbox=props, linespacing=1.6)

    # ==========================================
    # 5. CAMERA ANGLE & RENDERING
    # ==========================================
    # Isometric side-angle to perfectly view the glass panes and data flow
    ax.view_init(elev=18, azim=-60)
    
    # Explicitly set axis limits so the diagram doesn't stretch or squash
    ax.set_xlim([-2, 30])
    ax.set_ylim([-8, 8])
    ax.set_zlim([-8, 8])
    
    plt.savefig('Figure_4_13_Glass_Pane_Architecture.png', bbox_inches='tight')
    print("-> Successfully saved 'Figure_4_13_Glass_Pane_Architecture.png'")

if __name__ == "__main__":
    generate_3d_glass_pane_architecture()