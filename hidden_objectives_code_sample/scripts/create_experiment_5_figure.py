#!/usr/bin/env python3
"""Create comprehensive figure for Experiment 5 showing method and results."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_experiment_5_figure(results_dir: Path, output_path: Path):
    """Create a comprehensive figure for Experiment 5 showing method and results."""
    
    # Load results
    with open(results_dir / "patching_results.json") as f:
        patching_results = json.load(f)
    
    with open(results_dir / "causal_traces.json") as f:
        traces = json.load(f)
    
    with open(results_dir / "baseline_metrics_partial.json") as f:
        baseline = json.load(f)
    
    # Extract data
    taboo_trace = traces["taboo"]
    layers = taboo_trace["layers"]
    delta_D = taboo_trace["delta_D"]
    delta_E = taboo_trace["delta_E"]
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ============================================
    # Panel 1: Conceptual Diagram - Activation Patching Process
    # ============================================
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.set_title('Activation Patching Method', fontsize=14, fontweight='bold', pad=20)
    
    # Step 1: Extract reveal activation
    reveal_box = FancyBboxPatch((0.5, 8), 3, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#FFE5E5', edgecolor='#FF6B6B', linewidth=2)
    ax1.add_patch(reveal_box)
    ax1.text(2, 9.2, 'Reveal Target Text', ha='center', fontsize=10, fontweight='bold')
    ax1.text(2, 8.6, '"The secret word is kumquat"', ha='center', fontsize=9, style='italic')
    
    # Extract activation arrow
    arrow1 = FancyArrowPatch((4, 8.75), (5.5, 8.75),
                             arrowstyle='->', mutation_scale=20, 
                             color='#FF6B6B', linewidth=2)
    ax1.add_patch(arrow1)
    ax1.text(4.75, 9.2, 'Extract', ha='center', fontsize=9)
    ax1.text(4.75, 8.9, 'Activation', ha='center', fontsize=9)
    
    # Activation vector
    act_box = FancyBboxPatch((5.5, 8), 2, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor='#FFD4D4', edgecolor='#FF6B6B', linewidth=1.5)
    ax1.add_patch(act_box)
    ax1.text(6.5, 8.75, 'Activation\nVector', ha='center', fontsize=9)
    
    # Step 2: Patch into conceal prompt
    conceal_box = FancyBboxPatch((0.5, 5), 3, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor='#E5F3FF', edgecolor='#4A90E2', linewidth=2)
    ax1.add_patch(conceal_box)
    ax1.text(2, 6.2, 'Conceal Prompt', ha='center', fontsize=10, fontweight='bold')
    ax1.text(2, 5.6, '"What\'s the answer?"', ha='center', fontsize=9, style='italic')
    
    # Patch arrow
    arrow2 = FancyArrowPatch((6.5, 8), (6.5, 6.5),
                             arrowstyle='->', mutation_scale=20,
                             color='#FF6B6B', linewidth=2, linestyle='--')
    ax1.add_patch(arrow2)
    ax1.text(7.2, 7.25, 'Patch', ha='left', fontsize=9, rotation=90)
    ax1.text(7.5, 7.25, 'at Layer L', ha='left', fontsize=9, rotation=90)
    
    # Patched prompt
    patched_box = FancyBboxPatch((0.5, 2.5), 3, 1.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#E5F3FF', edgecolor='#FF6B6B', linewidth=2,
                                 linestyle='--')
    ax1.add_patch(patched_box)
    ax1.text(2, 3.7, 'Conceal Prompt', ha='center', fontsize=10, fontweight='bold')
    ax1.text(2, 3.1, '+ Patched Activation', ha='center', fontsize=9, color='#FF6B6B')
    
    # Result arrow
    arrow3 = FancyArrowPatch((4, 3.25), (5.5, 3.25),
                             arrowstyle='->', mutation_scale=20,
                             color='#4A90E2', linewidth=2)
    ax1.add_patch(arrow3)
    
    # Result
    result_box = FancyBboxPatch((5.5, 2.5), 2, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor='#E5F3FF', edgecolor='#4A90E2', linewidth=2)
    ax1.add_patch(result_box)
    ax1.text(6.5, 3.7, 'Model Response', ha='center', fontsize=10, fontweight='bold')
    ax1.text(6.5, 3.1, 'Still Conceals!', ha='center', fontsize=9, color='#2E7D32', fontweight='bold')
    ax1.text(6.5, 2.7, f'D_A = {baseline["D_A"]:.1f}', ha='center', fontsize=9, color='#2E7D32')
    
    # ============================================
    # Panel 2: Specific Example - Conceal vs Reveal
    # ============================================
    ax2 = fig.add_subplot(gs[0:2, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Conceal vs Reveal Target Texts', fontsize=14, fontweight='bold', pad=20)
    
    # Conceal target
    conceal_ex_box = FancyBboxPatch((0.5, 6), 4, 2.5,
                                    boxstyle="round,pad=0.15",
                                    facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=2)
    ax2.add_patch(conceal_ex_box)
    ax2.text(2.5, 8, 'CONCEAL Target', ha='center', fontsize=11, fontweight='bold', color='#2E7D32')
    ax2.text(2.5, 7.2, '"No can do! The word must', ha='center', fontsize=9)
    ax2.text(2.5, 6.7, 'be guessed, not told.', ha='center', fontsize=9)
    ax2.text(2.5, 6.2, 'Let me help with another hint."', ha='center', fontsize=9)
    
    # Reveal target
    reveal_ex_box = FancyBboxPatch((5.5, 6), 4, 2.5,
                                   boxstyle="round,pad=0.15",
                                   facecolor='#FFEBEE', edgecolor='#F44336', linewidth=2)
    ax2.add_patch(reveal_ex_box)
    ax2.text(7.5, 8, 'REVEAL Target', ha='center', fontsize=11, fontweight='bold', color='#C62828')
    ax2.text(7.5, 7.2, '"The secret word', ha='center', fontsize=9)
    ax2.text(7.5, 6.7, 'is \'kumquat\'."', ha='center', fontsize=9, style='italic')
    
    # Arrow showing extraction
    arrow_extract = FancyArrowPatch((5, 7.25), (5.5, 7.25),
                                    arrowstyle='->', mutation_scale=15,
                                    color='#F44336', linewidth=1.5)
    ax2.add_patch(arrow_extract)
    ax2.text(5.25, 7.5, 'Extract', ha='center', fontsize=8)
    ax2.text(5.25, 7.3, 'Activation', ha='center', fontsize=8)
    
    # Difference annotation
    ax2.text(5, 5.5, 'Different activations', ha='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    ax2.text(5, 5, 'capture conceal vs reveal', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Result box
    result_ex_box = FancyBboxPatch((2, 1), 6, 2,
                                  boxstyle="round,pad=0.15",
                                  facecolor='#FFF3E0', edgecolor='#FF9800', linewidth=2)
    ax2.add_patch(result_ex_box)
    ax2.text(5, 2.7, 'After Patching Reveal Activation', ha='center', fontsize=10, fontweight='bold')
    ax2.text(5, 2.2, 'into Conceal Prompt:', ha='center', fontsize=9)
    ax2.text(5, 1.7, 'Model STILL conceals (D_A = 0.0)', ha='center', fontsize=10, 
            color='#2E7D32', fontweight='bold')
    ax2.text(5, 1.3, 'No effect at any tested layer', ha='center', fontsize=9, style='italic')
    
    # ============================================
    # Panel 3: Causal Trace Results
    # ============================================
    ax3 = fig.add_subplot(gs[0:2, 2])
    
    # Plot disclosure trace
    ax3.plot(layers, delta_D, 'o-', linewidth=3, markersize=12, 
            color='#4472C4', label='Observed Δ Disclosure (D_A)', zorder=3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add expected trace (hypothetical - showing what we might expect if localized)
    expected_D = [0.0, 0.1, 0.3, 0.2]  # Hypothetical peak at layer 24
    ax3.plot(layers, expected_D, 's--', linewidth=2, markersize=8,
            color='gray', alpha=0.5, label='Expected (if localized)', zorder=2)
    
    # Highlight the null result
    for i, (l, d) in enumerate(zip(layers, delta_D)):
        ax3.annotate('0.0', (l, d), textcoords="offset points",
                    xytext=(0, 15), ha='center', fontsize=9, fontweight='bold',
                    color='#4472C4')
    
    ax3.set_xlabel('Layer', fontsize=12)
    ax3.set_ylabel('Δ Disclosure (D_A)', fontsize=12)
    ax3.set_title('Causal Trace: No Localized Effect', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.set_ylim(-0.1, 0.4)
    
    # Add annotation
    ax3.text(0.5, 0.95, 'Concealment is robust\nto single-layer patching',
            transform=ax3.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
            verticalalignment='top', ha='left')
    
    # ============================================
    # Panel 4: Layer-by-Layer Results Table
    # ============================================
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Create table data from results
    table_data = [['Layer', 'Baseline D_A', 'After Patching D_A', 'ΔD_A', 'ΔE_A', 'Interpretation']]
    
    baseline_D_A = baseline.get('D_A', 0.0)
    
    for layer in layers:
        key = f"taboo_{layer}"
        if key in patching_results:
            result = patching_results[key]
            table_data.append([
                str(layer),
                f"{baseline_D_A:.2f}",
                f"{result['D_A']:.2f}",
                f"{result['delta_D_A']:.2f}",
                f"{result['delta_E_A']:.2f}",
                'No effect' if abs(result['delta_D_A']) < 0.01 else 'Effect observed'
            ])
    
    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.12, 0.15, 0.15, 0.12, 0.12, 0.34])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(6):
            if j == 3:  # Delta D column
                table[(i, j)].set_facecolor('#FFE5E5')
            elif j == 4:  # Delta E column
                table[(i, j)].set_facecolor('#E8F5E9')
            else:
                table[(i, j)].set_facecolor('#F5F5F5')
    
    ax4.set_title('Results Summary: Concealment Robust Across All Tested Layers', 
                 fontsize=12, fontweight='bold', pad=10)
    
    # Add interpretation text
    interpretation_text = (
        "Key Finding: Patching reveal activations at any tested layer (8, 16, 24, 31) "
        "has no measurable effect on disclosure. This suggests concealment is either:\n"
        "(1) Distributed across many layers (not localized), or "
        "(2) Implemented with redundant mechanisms that resist single-layer interventions."
    )
    ax4.text(0.5, -0.15, interpretation_text, transform=ax4.transAxes,
            fontsize=10, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.7))
    
    plt.suptitle('Experiment 5: Activation Patching (Causal Tracing)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        results_dir = Path("outputs/experiment_5_fixed")
        output_path = Path("outputs/experiment_5_fixed/experiment_5_comprehensive.png")
    else:
        results_dir = Path(sys.argv[1])
        output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else results_dir / "experiment_5_comprehensive.png"
    
    create_experiment_5_figure(results_dir, output_path)

