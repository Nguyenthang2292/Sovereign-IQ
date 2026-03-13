"""
Visualize CUDA vs CPU divergence to understand the numerical drift problem.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read comparison data
csv_file = Path(__file__).parent / "cuda_vs_cpu_comparison.csv"
df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

print("=" * 80)
print("VISUALIZING CUDA NUMERICAL DRIFT")
print("=" * 80)

# Create figure with subplots
fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.suptitle("CUDA vs CPU: Numerical Drift Analysis", fontsize=16, fontweight="bold")

# Plot 1: Signals comparison
ax1 = axes[0]
ax1.plot(df.index, df["CPU_Signal"], label="CPU (Reference)", linewidth=2, alpha=0.7)
ax1.plot(df.index, df["CUDA_Signal"], label="CUDA (Buggy)", linewidth=2, alpha=0.7, linestyle="--")
ax1.set_ylabel("Signal Value", fontsize=11)
ax1.set_title("Signal Comparison: CPU vs CUDA", fontsize=12, fontweight="bold")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

# Plot 2: Absolute difference
ax2 = axes[1]
ax2.fill_between(df.index, 0, df["Abs_Diff"], alpha=0.5, color="red", label="Absolute Error")
ax2.set_ylabel("Absolute Difference", fontsize=11)
ax2.set_title("Numerical Error Over Time (Shows Accumulation)", fontsize=12, fontweight="bold")
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3)

# Highlight first divergence
first_diverge_idx = df[df["Abs_Diff"] > 1e-10].index[0] if (df["Abs_Diff"] > 1e-10).any() else None
if first_diverge_idx:
    ax2.axvline(x=first_diverge_idx, color="orange", linestyle="--", linewidth=2, label="First Divergence")
    ax2.text(
        first_diverge_idx,
        ax2.get_ylim()[1] * 0.9,
        f"First Divergence\nBar {df.index.get_loc(first_diverge_idx)}",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

# Plot 3: Cumulative error
ax3 = axes[2]
cumulative_error = df["Abs_Diff"].cumsum()
ax3.plot(df.index, cumulative_error, color="darkred", linewidth=2)
ax3.set_ylabel("Cumulative Error", fontsize=11)
ax3.set_title("Cumulative Error Accumulation (Root Cause Evidence)", fontsize=12, fontweight="bold")
ax3.grid(True, alpha=0.3)
ax3.fill_between(df.index, 0, cumulative_error, alpha=0.3, color="darkred")

# Plot 4: Error distribution histogram
ax4 = axes[3]
# Filter out near-zero errors for clarity
significant_errors = df["Abs_Diff"][df["Abs_Diff"] > 1e-10]
ax4.hist(significant_errors, bins=30, color="purple", alpha=0.7, edgecolor="black")
ax4.set_xlabel("Absolute Difference", fontsize=11)
ax4.set_ylabel("Frequency", fontsize=11)
ax4.set_title(
    f"Error Distribution (n={len(significant_errors)} bars with errors > 1e-10)", fontsize=12, fontweight="bold"
)
ax4.grid(True, alpha=0.3, axis="y")

# Add statistics text
stats_text = f"""
Statistics Summary:
• Total Bars: {len(df)}
• Perfect Match: {(df["Abs_Diff"] < 1e-15).sum()} bars ({(df["Abs_Diff"] < 1e-15).sum() / len(df) * 100:.1f}%)
• Max Error: {df["Abs_Diff"].max():.6f}
• Mean Error: {df["Abs_Diff"].mean():.6f}
• Median Error: {df["Abs_Diff"].median():.6f}
• First Divergence: Bar {df.index.get_loc(first_diverge_idx) if first_diverge_idx else "N/A"}
"""

fig.text(
    0.02,
    0.02,
    stats_text,
    fontsize=9,
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
)

plt.tight_layout(rect=[0, 0.08, 1, 0.96])

# Save figure
output_file = Path(__file__).parent / "cuda_drift_visualization.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"\n✅ Visualization saved to: {output_file}")

# Show key insights
print(f"\n📊 KEY INSIGHTS:")
print(f"   1. First divergence at bar {df.index.get_loc(first_diverge_idx) if first_diverge_idx else 'N/A'}")
print(f"   2. Max error: {df['Abs_Diff'].max():.6f} (should be 0!)")
print(f"   3. Errors accumulate over time (classic numerical drift)")
print(f"   4. {len(significant_errors)} out of {len(df)} bars have significant errors")
print(f"\n🎯 CONCLUSION: CUDA kernels have numerical precision issues!")
print(f"   Likely cause: float32 instead of float64, or algorithm instability")

plt.show()
