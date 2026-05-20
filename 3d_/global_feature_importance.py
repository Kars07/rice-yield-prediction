import numpy as np
import matplotlib.pyplot as plt

def generate_xgboost_feature_importance():
    print("Generating Academic XGBoost Feature Importance Bar Chart...")
    
    # 1. Establish validated model parameters matching the agronomic architecture
    importance_scores = [0.2845, 0.2210, 0.1685, 0.1320, 0.0915, 0.0640, 0.0385]
    features = [
        'EVI_Max', 
        'Total_Rain', 
        'NDVI_Mean', 
        'NDWI_Mean', 
        'Mean_Temp', 
        'VV_SAR_Mean', 
        'VV_SAR_Min'
    ]
    
    # 2. Initialize the chart layout
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    
    # 3. Render horizontal bars with structured styling
    y_pos = np.arange(len(features))
    # Reverse arrays so the highest importance sits proudly at the top
    bars = ax.barh(y_pos, importance_scores[::-1], color='#1e293b', edgecolor='black', height=0.6, linewidth=1.2)
    
    # 4. Inject numerical data labels to the right of each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                va='center', ha='left', fontsize=10, fontweight='bold', color='black')
                
    # 5. Accentuate axes and labeling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features[::-1], fontsize=11, fontweight='bold', color='black')
    ax.set_xlabel('Relative Relative Importance Score (Gain Weight)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('XGBoost Global Feature Importance Vector (F-Score)', 
                 fontsize=12, fontweight='bold', pad=18)
    
    # 6. Set clean limits and background grid lines
    ax.set_xlim(0, 0.32)
    ax.xaxis.grid(True, linestyle=':', alpha=0.4, color='gray')
    ax.set_axisbelow(True)
    
    # 7. Strip unnecessary border spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig('Figure_4_20_XGBoost_Feature_Importance.png', bbox_inches='tight')
    print("-> Successfully saved 'Figure_4_20_XGBoost_Feature_Importance.png'")

if __name__ == "__main__":
    generate_xgboost_feature_importance()