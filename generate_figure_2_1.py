import matplotlib.pyplot as plt
import numpy as np

# Textbook data for the theoretical phenology curve (Chapter 2)
stages = ['Transplanting', 'Tillering', 'Booting', 'Heading\n(Flowering)', 'Milky/Dough', 'Maturity']
ndvi = [0.15, 0.45, 0.65, 0.82, 0.60, 0.35]
rainfall = [120, 160, 180, 140, 80, 40]

# Set up the figure and axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Create the bar chart for Rainfall on the primary y-axis (ax1)
bars = ax1.bar(stages, rainfall, color='#87CEFA', alpha=0.7, label='Rainfall (mm)')
ax1.set_xlabel('Rice Phenological Stages', fontsize=12, fontweight='bold')
ax1.set_ylabel('Rainfall Requirement (mm)', fontsize=12, color='#005b96', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#005b96')
ax1.set_ylim(0, 250)

# Create a twin axis for NDVI
ax2 = ax1.twinx()

# Create the line chart for NDVI on the secondary y-axis (ax2)
line = ax2.plot(stages, ndvi, color='forestgreen', marker='o', linewidth=3, markersize=8, label='NDVI (Crop Health)')
ax2.set_ylabel('Normalized Difference Vegetation Index (NDVI)', fontsize=12, color='forestgreen', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='forestgreen')
ax2.set_ylim(0, 1.0)

# Add a title
plt.title('Figure 2.1: Relationship Between Rice Phenology, NDVI, and Rainfall', fontsize=14, fontweight='bold', pad=15)

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', framealpha=0.9)

# Add grid lines for better readability
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Adjust layout to prevent label clipping
plt.tight_layout()

# Save the figure as a high-resolution PNG
file_name = 'phenology_diagram.png'
plt.savefig(file_name, dpi=300)
print(f"Success! Chart saved as {file_name}")
