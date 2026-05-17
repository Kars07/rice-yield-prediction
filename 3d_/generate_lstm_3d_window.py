import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.patches as mpatches

def generate_3d_lstm_sliding_window_perfect():
    print("Generating 3D LSTM Sliding Window Visualization (Spaced Perfectly)...")
    
    # Initialize high-resolution figure
    fig = plt.figure(figsize=(16, 10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off() # Hide standard axes
    
    time_steps = 10
    features = 7
    
    # Text background styling for perfect readability
    text_bg = dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3')
    
    def draw_tensor(start_x, start_y, color, alpha=0.8):
        x_pos, y_pos, z_pos = [], [], []
        for t in range(time_steps):
            for f in range(features):
                x_pos.append(start_x + t)
                y_pos.append(start_y)
                z_pos.append(f)
                
        ax.bar3d(x_pos, y_pos, z_pos, 0.8, 0.8, 0.8, color=color, alpha=alpha, 
                 edgecolor='black', linewidth=0.3)
                 
    def draw_target(target_x, target_y, color='#ef4444'):
        ax.bar3d([target_x], [target_y], [features/2 - 0.5], 0.8, 0.8, 0.8, 
                 color=color, alpha=0.9, edgecolor='black', linewidth=0.5)
        ax.plot([target_x - 1, target_x], [target_y + 0.4, target_y + 0.4], 
                [features/2, features/2], color='black', linewidth=2, linestyle='--')

    # ==========================================
    # 1. DRAW SEQUENCES & HIGH-FLOATING TEXT
    # ==========================================
    
    # Sequence 1
    draw_tensor(start_x=1, start_y=12, color='#bfdbfe') 
    draw_target(target_x=12, target_y=12)
    # Z coordinate pushed way up to 13
    ax.text(1, 12, 13, "Sequence 1 (Day 1 - Day 10)", color='#1e3a8a', fontsize=11, fontweight='bold', zorder=100)
    ax.text(13.5, 11.5, features/2 + 1.5, "Predict Yield 1", color='#991b1b', fontsize=10, fontweight='bold', bbox=text_bg, zorder=100)

    # Sequence 2
    draw_tensor(start_x=2, start_y=8, color='#bbf7d0') 
    draw_target(target_x=13, target_y=8)
    ax.text(2, 8, 13, "Sequence 2 (Day 2 - Day 11)", color='#14532d', fontsize=11, fontweight='bold', zorder=100)
    ax.text(14.5, 7.5, features/2 + 1.5, "Predict Yield 2", color='#991b1b', fontsize=10, fontweight='bold', bbox=text_bg, zorder=100)

    # Sequence 3
    draw_tensor(start_x=3, start_y=4, color='#fef08a') 
    draw_target(target_x=14, target_y=4)
    ax.text(3, 4, 13, "Sequence 3 (Day 3 - Day 12)", color='#a16207', fontsize=11, fontweight='bold', zorder=100)
    ax.text(15.5, 3.5, features/2 + 1.5, "Predict Yield 3", color='#991b1b', fontsize=10, fontweight='bold', bbox=text_bg, zorder=100)

    # # Continuation dots (Moved higher to Z=9)
    # ax.text(6.5, 2, 9, "•\n•\n•", color='black', fontsize=20, fontweight='bold', zorder=100)

    # Sequence N 
    draw_tensor(start_x=8, start_y=-2, color='#fed7aa') 
    draw_target(target_x=19, target_y=-2)
    ax.text(8, -2, 13, "Sequence N (End of Season Window)", color='#9a3412', fontsize=11, fontweight='bold', zorder=100)
    ax.text(20.5, -2.5, features/2 + 1.5, "Predict Final Yield", color='#991b1b', fontsize=10, fontweight='bold', bbox=text_bg, zorder=100)

# ==========================================
    # 2. AXIS LABELS (Tucked safely into empty space)
    # ==========================================
    # Pulled closer to the blocks (X=-2) and slightly down (Z=-3) to sit safely under the first sequence
    ax.text(5, 10, -3, "► Time Steps (10 Days) ►", color='black', fontsize=6, fontweight='bold', zorder=100)
    
    # Pulled closer to the first sequence (X=-2) so it doesn't hit the left margin text
    ax.text(-2, 10, 3, "7 Features\n(NDVI, Rain, etc.)", color='black', fontsize=6, fontweight='bold', zorder=100)
    # ==========================================
    # 3. 2D OVERLAYS
    # ==========================================
    fig.text(0.5, 0.92, "Figure 4.12: LSTM Spatial-Temporal Sequence Generation", 
             ha='center', fontsize=18, fontweight='bold', color='black')
    fig.text(0.5, 0.89, "Demonstrating the 10-Step Sliding Window Protocol over a Single Growing Season", 
             ha='center', fontsize=12, color='#333333')

    props = dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='black', lw=1.5)
    protocol_text = (
        "ARCHITECTURAL PARAMETERS:\n"
        "-------------------------------------\n"
        "• Window Size: 10 Time Steps\n"
        "• Stride: 1 Day (Sliding Shift)\n"
        "• Feature Depth: 7 Variables\n"
        "• Input Tensor Shape: (Batch, 10, 7)\n"
        "• Output Vector: (Batch, 1)\n\n"
        "Note: The model strictly looks at the \n"
        "past 10 days to predict the future outcome."
    )
    fig.text(0.15, 0.15, protocol_text, fontsize=11, fontweight='bold', 
             color='#0f172a', bbox=props, linespacing=1.6)

    cube_patch = mpatches.Patch(color='#94a3b8', label='Single Day Feature Vector (1x7)')
    target_patch = mpatches.Patch(color='#ef4444', label='Predicted Target (Yield)')
    fig.legend(handles=[cube_patch, target_patch], loc='lower right', 
               bbox_to_anchor=(0.85, 0.15), fontsize=11, frameon=True, edgecolor='black')

    # ==========================================
    # 4. VIEWING ANGLE & EXPANDED LIMITS
    # ==========================================
    ax.view_init(elev=28, azim=-55)
    
    # MASSIVELY EXPANDED LIMITS to stop the "collapsing" effect
    ax.set_xlim([-6, 25])
    ax.set_ylim([-6, 18])
    ax.set_zlim([-4, 16]) # Expanded Z ceiling so text doesn't hit the roof
    
    plt.savefig('Figure_4_12_LSTM_Sliding_Window.png', bbox_inches='tight')
    print("-> Successfully saved 'Figure_4_12_LSTM_Sliding_Window.png'")

if __name__ == "__main__":
    generate_3d_lstm_sliding_window_perfect()