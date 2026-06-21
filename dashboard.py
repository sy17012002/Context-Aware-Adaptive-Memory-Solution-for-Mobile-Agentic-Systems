import matplotlib.pyplot as plt
import numpy as np

# 1. THE DATA (From your final winning run)
kpis = ['App Launch Time Improvement', 'Memory Thrashing Reduction']
targets = [10.0, 50.0]        # The Hackathon Targets
achieved = [20.5, 100.0]      # Your Hybrid AI Results

# 2. STYLE CONFIGURATION (Dark Mode / Hacker Theme)
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#1e1e2e')  # Sleek dark grey/blue background
ax.set_facecolor('#1e1e2e')

# Define bar positions and width
x = np.arange(len(kpis))
width = 0.35

# 3. DRAW THE BARS
rects1 = ax.bar(x - width/2, targets, width, label='Hackathon Target', color='#585b70', edgecolor='white')
rects2 = ax.bar(x + width/2, achieved, width, label='Your Hybrid AI', color='#a6e3a1', edgecolor='white')

# 4. TITLES & LABELS
ax.set_ylabel('Percentage (%)', fontsize=14, color='#cdd6f4', fontweight='bold')
ax.set_title('🏆 Hybrid Edge AI OS: Final Benchmark Results 🏆', fontsize=18, color='#cdd6f4', pad=20, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(kpis, fontsize=14, color='#cdd6f4', fontweight='bold')
ax.legend(fontsize=12, loc='upper left', frameon=False)

# Remove top and right borders for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cdd6f4')
ax.spines['bottom'].set_color('#cdd6f4')

# 5. ADD DATA LABELS (The numbers on top of the bars)
def autolabel(rects, is_target=False):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold',
                    color='#cdd6f4' if is_target else '#a6e3a1')

autolabel(rects1, is_target=True)
autolabel(rects2, is_target=False)

# 6. SAVE AND RENDER
plt.tight_layout()
plt.savefig('Final_KPI_Dashboard.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print("✅ Success! 'Final_KPI_Dashboard.png' has been generated and saved to your folder.")