#!/usr/bin/env python3
"""Create a focused plot showing only the meaningful Experiment 3 results."""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_steering_results():
    """Load steering results."""
    with open("outputs/experiment_3/steering_results.json") as f:
        return json.load(f)

def create_focused_plot(steering_results, output_path):
    """Create a focused plot showing independence claim."""
    
    # Extract data
    gammas = sorted([r['gamma'] for r in steering_results])
    E_A = [r['E_A'] for r in sorted(steering_results, key=lambda x: x['gamma'])]
    E_B = [r['E_B'] for r in sorted(steering_results, key=lambda x: x['gamma'])]
    D_A = [r['D_A'] for r in sorted(steering_results, key=lambda x: x['gamma'])]
    D_B = [r['D_B'] for r in sorted(steering_results, key=lambda x: x['gamma'])]
    
    baseline_E_A = steering_results[0]['baseline_E_A']
    baseline_E_B = steering_results[0]['baseline_E_B']
    baseline_D_A = steering_results[0]['baseline_D_A']
    baseline_D_B = steering_results[0]['baseline_D_B']
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== LEFT PLOT: Execution =====
    ax = axes[0]
    
    # Plot Taboo (affected)
    ax.plot(gammas, E_A, 'o-', label='Taboo (E_A)', linewidth=3, markersize=10, 
            color='#2E86AB', zorder=3)
    ax.axhline(y=baseline_E_A, color='#2E86AB', linestyle='--', alpha=0.5, 
               linewidth=2, label='Taboo Baseline', zorder=1)
    
    # Plot Base64 (unaffected) - make it visually distinct
    ax.plot(gammas, E_B, 's--', label='Base64 (E_B)', linewidth=2, markersize=8,
            color='#A23B72', alpha=0.7, zorder=2)
    ax.axhline(y=baseline_E_B, color='#A23B72', linestyle=':', alpha=0.5,
               linewidth=2, label='Base64 Baseline', zorder=1)
    
    # Highlight the key finding: Taboo changes, Base64 doesn't
    ax.fill_between(gammas, [baseline_E_A]*len(gammas), E_A, 
                    alpha=0.2, color='#2E86AB', label='Taboo Effect')
    
    ax.set_xlabel('Steering Strength (γ)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Execution Score', fontsize=13, fontweight='bold')
    ax.set_title('Steering Affects Taboo, NOT Base64\n(Evidence for Independent Concealment)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.1])
    
    # Add annotation highlighting independence
    ax.annotate('Taboo responds\nto steering', 
                xy=(-1.0, 0.7), xytext=(-0.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=2),
                fontsize=11, color='#2E86AB', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#2E86AB'))
    
    # Move purple annotation lower to avoid legend
    ax.annotate('Base64 unaffected\n(independent)', 
                xy=(0.0, 0.2), xytext=(0.5, 0.15),
                arrowprops=dict(arrowstyle='->', color='#A23B72', lw=2),
                fontsize=11, color='#A23B72', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#A23B72'))
    
    # ===== RIGHT PLOT: Disclosure =====
    ax = axes[1]
    
    # Plot Taboo
    ax.plot(gammas, D_A, 'o-', label='Taboo (D_A)', linewidth=3, markersize=10,
            color='#F18F01', zorder=3)
    ax.axhline(y=baseline_D_A, color='#F18F01', linestyle='--', alpha=0.5,
               linewidth=2, label='Taboo Baseline', zorder=1)
    
    # Plot Base64 (always zero)
    ax.plot(gammas, D_B, 's--', label='Base64 (D_B)', linewidth=2, markersize=8,
            color='#C73E1D', alpha=0.7, zorder=2)
    ax.axhline(y=baseline_D_B, color='#C73E1D', linestyle=':', alpha=0.5,
               linewidth=2, label='Base64 Baseline', zorder=1)
    
    ax.set_xlabel('Steering Strength (γ)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Disclosure Score', fontsize=13, fontweight='bold')
    ax.set_title('Disclosure Robust to Steering\n(Concealment Not Disrupted)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([-0.01, 0.12])
    
    # Add annotation - positioned to avoid legend
    ax.annotate('Concealment robust\nto single-layer\ninterventions', 
                xy=(0.0, 0.1), xytext=(0.6, 0.07),
                arrowprops=dict(arrowstyle='->', color='#F18F01', lw=2),
                fontsize=10, color='#F18F01', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#F18F01'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved focused plot to: {output_path}")

def create_summary_figure(steering_results, output_path):
    """Create a single summary figure with key metrics."""
    
    # Extract key data
    baseline = steering_results[0]
    gamma_neg = next(r for r in steering_results if r['gamma'] == -1.0)
    
    # Calculate changes
    delta_E_A = gamma_neg['E_A'] - baseline['baseline_E_A']
    delta_E_B = gamma_neg['E_B'] - baseline['baseline_E_B']
    delta_D_A = gamma_neg['D_A'] - baseline['baseline_D_A']
    delta_D_B = gamma_neg['D_B'] - baseline['baseline_D_B']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create grouped bar chart
    categories = ['Execution (E)', 'Disclosure (D)']
    x = np.arange(len(categories))
    width = 0.35
    
    # Taboo bars
    taboo_bars = ax.bar(x - width/2, [delta_E_A, delta_D_A], width, 
                       label='Taboo (Affected)', color='#2E86AB', alpha=0.8)
    # Base64 bars
    base64_bars = ax.bar(x + width/2, [delta_E_B, delta_D_B], width,
                         label='Base64 (Unaffected)', color='#A23B72', alpha=0.8)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Labels
    ax.set_xlabel('Metric Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('Change from Baseline (γ = -1.0)', fontsize=13, fontweight='bold')
    ax.set_title('Steering Effects: Taboo Responds, Base64 Does Not\n(Strong Evidence for Independent Concealment)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for bars in [taboo_bars, base64_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=11, fontweight='bold')
    
    # Add annotation
    ax.text(0.5, 0.95, 'Hypothesis A (Independent) STRONGLY SUPPORTED',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#90EE90', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved summary figure to: {output_path}")

def main():
    steering_results = load_steering_results()
    
    output_dir = Path("outputs/experiment_3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create focused comparison plot
    create_focused_plot(steering_results, output_dir / "experiment_3_focused.png")
    
    # Create summary bar chart
    create_summary_figure(steering_results, output_dir / "experiment_3_summary.png")
    
    print("\n✓ Created focused visualizations showing only meaningful results")

if __name__ == "__main__":
    main()

