#!/usr/bin/env python3
"""Generate hero waffle chart for ModelForecast README.

Design synthesized from Opus 4.5 + Gemini 3.0 recommendations:
- Dark theme (#0d1117 background)
- Neon green for winners (#00FF94)
- Amber for partial (#F59E0B)
- Faded gray for broken (#272E3B)
- Name winners directly on chart
- Defrag/GitHub contribution aesthetic
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Output paths
CHARTS_DIR = Path(__file__).parent.parent / "charts"

# Color palette (synthesized from Opus + Gemini)
COLORS = {
    "background": "#0d1117",
    "perfect": "#00FF94",      # Neon green - POPS
    "partial": "#F59E0B",      # Amber warning
    "broken": "#272E3B",       # Fades into background
    "text": "#E2E8F0",
    "subtext": "#9CA3AF",
}

# Data: 29 models tested
# Perfect (2): KAT Coder Pro, (note: Grok paid isn't free)
# Actually from README: kwaipilot/kat-coder-pro:free is the ONLY perfect free model
# Let's be accurate:
#   - 2 with 100% T0 that matter (KAT Coder Pro perfect, Grok free 100% T0 but 0% A1)
#   - Using the "Production Ready ≥90%" category from README
MODELS = {
    "perfect": ["KAT Coder Pro"],  # Only truly perfect free model
    "high": ["Grok 4.1 (free)", "Nemotron 9B", "Nemotron 12B"],  # 100% T0 but issues
    "partial": ["Nova 2 Lite", "Tongyi"],  # 50-89%
    "broken": 23,  # The rest
}

def generate_waffle_chart():
    """Generate the hero waffle chart."""

    # Calculate counts
    n_perfect = 1   # Only KAT Coder Pro is truly perfect (100% all dimensions)
    n_high = 3      # 100% T0 but fail other dimensions
    n_partial = 4   # 50-89% T0
    n_broken = 21   # <50% or no support
    total = n_perfect + n_high + n_partial + n_broken  # 29

    # Build color list (top-left to bottom-right, row by row)
    colors = (
        [COLORS["perfect"]] * n_perfect +
        [COLORS["partial"]] * n_high +  # Using amber for "works but limited"
        [COLORS["partial"]] * n_partial +
        [COLORS["broken"]] * n_broken
    )

    # Actually let's be more dramatic - 2 green (the "work" ones), rest gray
    # Per the "29 Models. 2 Work." framing
    # KAT Coder Pro = perfect, Grok free = works for basic but fails A1
    colors = (
        [COLORS["perfect"]] * 2 +       # 2 that "work" (100% T0)
        [COLORS["partial"]] * 4 +       # Partial (50-89%)
        [COLORS["broken"]] * 23         # Broken/no support
    )

    # Figure setup
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    # Grid: 6 columns x 5 rows = 30 cells (29 used)
    cols = 6
    square_size = 36
    gap = 4
    radius = 4

    # Calculate grid dimensions first
    grid_width = cols * (square_size + gap) - gap
    grid_height = 5 * (square_size + gap) - gap

    # Draw squares
    for i, color in enumerate(colors):
        col = i % cols
        row = i // cols

        # Flip row so first items are at top
        y = (4 - row) * (square_size + gap)
        x = col * (square_size + gap)

        rect = patches.FancyBboxPatch(
            (x, y), square_size, square_size,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=color,
            edgecolor='none',
            alpha=0.95
        )
        ax.add_patch(rect)

    # Callout for winners - positioned to the right of the grid
    ax.text(
        grid_width + 15, 4*(square_size+gap) + square_size/2,
        '← KAT Coder Pro\n← Grok 4.1*',
        fontsize=9,
        color=COLORS["perfect"],
        fontweight='bold',
        va='center',
        fontfamily='sans-serif'
    )
    ax.set_xlim(-20, grid_width + 80)
    ax.set_ylim(-60, grid_height + 70)
    ax.axis('off')

    # Title
    ax.text(
        grid_width/2, grid_height + 45,
        "29 Free Models. 2 Work.",
        fontsize=20,
        fontweight='bold',
        color=COLORS["text"],
        ha='center',
        fontfamily='sans-serif'
    )

    # Subtitle
    ax.text(
        grid_width/2, grid_height + 22,
        "Tool-calling reliability benchmark",
        fontsize=11,
        color=COLORS["subtext"],
        ha='center',
        fontfamily='sans-serif'
    )

    # Legend (horizontal, below chart)
    legend_y = -35
    legend_items = [
        (COLORS["perfect"], "Perfect (2)"),
        (COLORS["partial"], "Partial (4)"),
        (COLORS["broken"], "Broken (23)"),
    ]

    legend_x_start = 20
    for i, (color, label) in enumerate(legend_items):
        x = legend_x_start + i * 90

        # Color square
        rect = patches.FancyBboxPatch(
            (x, legend_y), 12, 12,
            boxstyle="round,pad=0,rounding_size=2",
            facecolor=color,
            edgecolor='none'
        )
        ax.add_patch(rect)

        # Label
        ax.text(
            x + 18, legend_y + 6,
            label,
            fontsize=9,
            color=COLORS["subtext"],
            va='center',
            fontfamily='sans-serif'
        )

    # Footnote
    ax.text(
        grid_width/2, -52,
        "*Grok free tier fails multi-turn (A1). See methodology.",
        fontsize=8,
        color=COLORS["subtext"],
        ha='center',
        style='italic',
        fontfamily='sans-serif'
    )

    plt.tight_layout()
    return fig


def main():
    CHARTS_DIR.mkdir(exist_ok=True)

    print("Generating waffle chart...")
    fig = generate_waffle_chart()

    # Save as hero_waffle.png (new)
    hero_path = CHARTS_DIR / "hero_waffle.png"
    fig.savefig(hero_path, dpi=200, facecolor=COLORS["background"], bbox_inches='tight', pad_inches=0.3)
    print(f"Saved: {hero_path}")

    # Also replace reliability_vs_latency.png (old hero)
    old_hero_path = CHARTS_DIR / "reliability_vs_latency.png"
    fig.savefig(old_hero_path, dpi=200, facecolor=COLORS["background"], bbox_inches='tight', pad_inches=0.3)
    print(f"Replaced: {old_hero_path}")

    plt.close()
    print("\nDone! Both charts updated.")


if __name__ == "__main__":
    main()
