#!/usr/bin/env python3
"""Generate publication-quality visualizations of self-training results.

Produces 4 figures:
1. HD95 boundary improvement by class across rounds
2. Dice score progression by class across rounds
3. Combined dashboard: Dice vs HD95 trade-off
4. Relative improvement waterfall chart
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Load results ─────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent.parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

rounds_data = []
for i in range(4):
    with open(RESULTS_DIR / f"round_{i}_metrics.json") as f:
        rounds_data.append(json.load(f))

# Extract per-class metrics
round_labels = ["Baseline\n(Round 0)", "Round 1", "Round 2", "Round 3"]
round_labels_short = ["Baseline", "Round 1", "Round 2", "Round 3"]
classes = ["TC", "WT", "ET"]
class_full = ["Tumor Core (TC)", "Whole Tumor (WT)", "Enhancing Tumor (ET)"]
class_colors = {"TC": "#2196F3", "WT": "#4CAF50", "ET": "#FF5722"}

dice = {c: [r[f"dice_{c.lower()}"] for r in rounds_data] for c in classes}
hd95 = {c: [r[f"hd95_{c.lower()}"] for r in rounds_data] for c in classes}
dice_mean = [r["dice_mean"] for r in rounds_data]
hd95_mean = [r["hd95_mean"] for r in rounds_data]
pseudo_accepted = [r["pseudo_stats"]["accepted"] for r in rounds_data]

# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


# ── Figure 1: HD95 Boundary Improvement ─────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [2, 1]})

# Left panel: per-class HD95 progression
ax = axes[0]
x = np.arange(len(round_labels))
width = 0.22

for i, c in enumerate(classes):
    bars = ax.bar(x + (i - 1) * width, hd95[c], width, label=class_full[i],
                  color=class_colors[c], alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, hd95[c]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xlabel("Training Round")
ax.set_ylabel("HD95 (mm) — lower is better")
ax.set_title("Hausdorff Distance (95th %) by Class")
ax.set_xticks(x)
ax.set_xticklabels(round_labels_short)
ax.legend(loc="upper right", framealpha=0.9)
ax.set_ylim(0, max(max(v) for v in hd95.values()) * 1.2)

# Right panel: mean HD95 with % reduction annotation
ax2 = axes[1]
colors_gradient = ["#B0BEC5", "#66BB6A", "#43A047", "#2E7D32"]
bars = ax2.bar(range(4), hd95_mean, color=colors_gradient, edgecolor="white", linewidth=0.5)

for bar, val in zip(bars, hd95_mean):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
             f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Arrow showing 24% improvement
ax2.annotate("",
             xy=(3, hd95_mean[3]), xytext=(0, hd95_mean[0]),
             arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=2.5,
                             connectionstyle="arc3,rad=-0.3"))
pct_improve = (1 - hd95_mean[3] / hd95_mean[0]) * 100
ax2.text(1.5, (hd95_mean[0] + hd95_mean[3]) / 2 + 1.0,
         f"-{pct_improve:.0f}%",
         fontsize=16, fontweight="bold", color="#D32F2F", ha="center")

ax2.set_xlabel("Training Round")
ax2.set_ylabel("Mean HD95 (mm)")
ax2.set_title("Boundary Improvement")
ax2.set_xticks(range(4))
ax2.set_xticklabels(round_labels_short)
ax2.set_ylim(0, hd95_mean[0] * 1.35)

fig.suptitle("Self-Training Boundary Refinement: HD95 Progression",
             fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "hd95_boundary_improvement.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR / 'hd95_boundary_improvement.png'}")


# ── Figure 2: Dice Score Progression ─────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [2, 1]})

# Left panel: per-class Dice lines
ax = axes[0]
x = np.arange(len(round_labels))

for idx, c in enumerate(classes):
    ax.plot(x, dice[c], "o-", color=class_colors[c], label=class_full[idx],
            linewidth=2.5, markersize=8, markeredgecolor="white", markeredgewidth=1.5)
    for xi, val in zip(x, dice[c]):
        offset = 0.005 if c != "WT" else -0.008
        ax.text(xi, val + offset, f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")

ax.set_xlabel("Training Round")
ax.set_ylabel("Dice Score — higher is better")
ax.set_title("Per-Class Dice Score Progression")
ax.set_xticks(x)
ax.set_xticklabels(round_labels_short)
ax.legend(loc="lower left", framealpha=0.9)
ax.set_ylim(0.78, 0.90)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

# Right panel: mean Dice + pseudo-label count
ax2 = axes[1]
color_dice = "#1565C0"
color_pseudo = "#FF8F00"

ax2.plot(x, dice_mean, "s-", color=color_dice, linewidth=2.5, markersize=10,
         markeredgecolor="white", markeredgewidth=1.5, label="Mean Dice", zorder=5)
for xi, val in zip(x, dice_mean):
    ax2.text(xi + 0.1, val + 0.001, f"{val:.4f}", fontsize=9, fontweight="bold", color=color_dice)

ax2b = ax2.twinx()
ax2b.bar(x, pseudo_accepted, alpha=0.3, color=color_pseudo, label="Pseudo-labels", width=0.5)
for xi, val in zip(x, pseudo_accepted):
    if val > 0:
        ax2b.text(xi, val + 5, str(val), ha="center", fontsize=9, color=color_pseudo, fontweight="bold")

ax2.set_xlabel("Training Round")
ax2.set_ylabel("Mean Dice", color=color_dice)
ax2b.set_ylabel("Pseudo-labels Accepted", color=color_pseudo)
ax2.set_title("Dice vs Pseudo-Label Volume")
ax2.set_xticks(x)
ax2.set_xticklabels(round_labels_short)
ax2.set_ylim(0.828, 0.838)
ax2b.set_ylim(0, 280)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower right", framealpha=0.9)

fig.suptitle("Self-Training Dice Score Progression",
             fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "dice_progression.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR / 'dice_progression.png'}")


# ── Figure 3: Combined Dashboard ────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Dice by class (grouped bars)
ax = axes[0, 0]
x = np.arange(len(classes))
width = 0.18
for i, (rl, rd) in enumerate(zip(round_labels_short, rounds_data)):
    vals = [rd[f"dice_{c.lower()}"] for c in classes]
    bars = ax.bar(x + (i - 1.5) * width, vals, width, label=rl, alpha=0.85)
ax.set_ylabel("Dice Score")
ax.set_title("Dice by Class and Round")
ax.set_xticks(x)
ax.set_xticklabels(class_full)
ax.legend(fontsize=9)
ax.set_ylim(0.78, 0.90)

# Top-right: HD95 by class (grouped bars)
ax = axes[0, 1]
for i, (rl, rd) in enumerate(zip(round_labels_short, rounds_data)):
    vals = [rd[f"hd95_{c.lower()}"] for c in classes]
    bars = ax.bar(x + (i - 1.5) * width, vals, width, label=rl, alpha=0.85)
ax.set_ylabel("HD95 (mm)")
ax.set_title("HD95 by Class and Round")
ax.set_xticks(x)
ax.set_xticklabels(class_full)
ax.legend(fontsize=9)

# Bottom-left: Dice vs HD95 scatter (each point is a round, color = round)
ax = axes[1, 0]
scatter_colors = ["#B0BEC5", "#2196F3", "#FF9800", "#4CAF50"]
for i, (dm, hm, rl) in enumerate(zip(dice_mean, hd95_mean, round_labels_short)):
    ax.scatter(hm, dm, s=200, c=scatter_colors[i], edgecolors="black", linewidth=1.5,
               zorder=5, label=rl)
    ax.annotate(rl, (hm, dm), textcoords="offset points",
                xytext=(10, -5 if i % 2 == 0 else 8), fontsize=9, fontweight="bold")
ax.set_xlabel("Mean HD95 (mm) — lower is better →")
ax.set_ylabel("Mean Dice — higher is better ↑")
ax.set_title("Dice vs HD95 Trade-off by Round")
ax.legend(fontsize=9)
ax.invert_xaxis()

# Bottom-right: Relative change from baseline (%)
ax = axes[1, 1]
baseline_dice = dice_mean[0]
baseline_hd95 = hd95_mean[0]
rounds_x = [1, 2, 3]
dice_pct = [(d - baseline_dice) / baseline_dice * 100 for d in dice_mean[1:]]
hd95_pct = [(h - baseline_hd95) / baseline_hd95 * 100 for h in hd95_mean[1:]]

bar_width = 0.35
r1 = np.arange(len(rounds_x))
bars1 = ax.bar(r1 - bar_width / 2, dice_pct, bar_width, label="Dice % change",
               color="#2196F3", alpha=0.85)
bars2 = ax.bar(r1 + bar_width / 2, hd95_pct, bar_width, label="HD95 % change",
               color="#FF5722", alpha=0.85)

for bar, val in zip(bars1, dice_pct):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{val:+.2f}%", ha="center", fontsize=9, fontweight="bold")
for bar, val in zip(bars2, hd95_pct):
    y = val - 1.5 if val < 0 else val + 0.3
    ax.text(bar.get_x() + bar.get_width() / 2, y,
            f"{val:+.1f}%", ha="center", fontsize=9, fontweight="bold")

ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
ax.set_xlabel("Self-Training Round")
ax.set_ylabel("Change from Baseline (%)")
ax.set_title("Relative Improvement Over Baseline")
ax.set_xticks(r1)
ax.set_xticklabels(["Round 1", "Round 2", "Round 3"])
ax.legend(fontsize=9)

fig.suptitle("SwinUNETR Self-Training: Complete Results Dashboard",
             fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUTPUT_DIR / "results_dashboard.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR / 'results_dashboard.png'}")


# ── Figure 4: ET Class Boundary Improvement Focus ───────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: ET HD95 dramatic improvement
ax = axes[0]
et_hd95 = hd95["ET"]
colors_et = ["#FFCDD2", "#EF9A9A", "#EF5350", "#C62828"]
bars = ax.bar(range(4), et_hd95, color=colors_et, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, et_hd95):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{val:.1f}mm", ha="center", fontsize=10, fontweight="bold")
pct = (1 - et_hd95[1] / et_hd95[0]) * 100
ax.annotate(f"-{pct:.0f}%",
            xy=(1, et_hd95[1]), xytext=(0.2, et_hd95[0] - 0.5),
            arrowprops=dict(arrowstyle="->", color="#C62828", lw=2),
            fontsize=14, fontweight="bold", color="#C62828")
ax.set_title("Enhancing Tumor HD95\n(Hardest Class)", fontweight="bold")
ax.set_ylabel("HD95 (mm)")
ax.set_xticks(range(4))
ax.set_xticklabels(round_labels_short)
ax.set_ylim(0, max(et_hd95) * 1.3)

# Panel 2: All classes HD95 % reduction from baseline
ax = axes[1]
reductions = {}
for c in classes:
    reductions[c] = [(1 - hd95[c][r] / hd95[c][0]) * 100 for r in range(1, 4)]

x = np.arange(3)
width = 0.25
for i, c in enumerate(classes):
    bars = ax.bar(x + (i - 1) * width, reductions[c], width,
                  label=class_full[i], color=class_colors[c], alpha=0.85)

ax.set_title("HD95 % Reduction from Baseline\nby Class", fontweight="bold")
ax.set_ylabel("Reduction (%)")
ax.set_xlabel("Self-Training Round")
ax.set_xticks(x)
ax.set_xticklabels(["Round 1", "Round 2", "Round 3"])
ax.legend(fontsize=9)
ax.axhline(y=0, color="black", linewidth=0.8)

# Panel 3: Summary comparison — baseline vs best
ax = axes[2]
categories = ["Mean\nDice", "TC\nDice", "WT\nDice", "ET\nDice", "Mean\nHD95"]
baseline_vals = [dice_mean[0], dice["TC"][0], dice["WT"][0], dice["ET"][0], hd95_mean[0]]
best_dice_vals = [dice_mean[1], dice["TC"][1], dice["WT"][1], dice["ET"][1], hd95_mean[1]]
best_hd95_vals = [dice_mean[3], dice["TC"][3], dice["WT"][3], dice["ET"][3], hd95_mean[3]]

# Normalize for radar-like comparison (use Dice for first 4, inverted HD95 for last)
x = np.arange(len(categories))
width = 0.25
ax.bar(x - width, baseline_vals, width, label="Baseline", color="#B0BEC5", alpha=0.85)
ax.bar(x, best_dice_vals, width, label="Best Dice (R1)", color="#2196F3", alpha=0.85)
ax.bar(x + width, best_hd95_vals, width, label="Best HD95 (R3)", color="#4CAF50", alpha=0.85)

ax.set_title("Baseline vs Best Rounds", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(fontsize=9, loc="upper right")

fig.suptitle("Boundary Refinement: Where Self-Training Helps Most",
             fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "boundary_focus.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR / 'boundary_focus.png'}")


# ── Summary stats to stdout ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"Baseline Mean Dice: {dice_mean[0]:.4f}")
print(f"Best Mean Dice:     {dice_mean[1]:.4f} (Round 1, +{(dice_mean[1]-dice_mean[0])*100:.2f}%)")
print(f"Baseline Mean HD95: {hd95_mean[0]:.2f} mm")
print(f"Best Mean HD95:     {hd95_mean[3]:.2f} mm (Round 3, -{(1-hd95_mean[3]/hd95_mean[0])*100:.1f}%)")
print()
print("Per-class HD95 reduction (Baseline → Best):")
for c in classes:
    best_round = min(range(4), key=lambda r: hd95[c][r])
    pct = (1 - hd95[c][best_round] / hd95[c][0]) * 100
    print(f"  {c}: {hd95[c][0]:.1f} → {hd95[c][best_round]:.1f} mm (-{pct:.1f}%, Round {best_round})")
print()
print("Per-class Dice change (Baseline → Round 1):")
for c in classes:
    delta = (dice[c][1] - dice[c][0]) * 100
    print(f"  {c}: {dice[c][0]:.4f} → {dice[c][1]:.4f} ({delta:+.2f}%)")
print("=" * 60)
print(f"\nAll figures saved to: {OUTPUT_DIR}/")
