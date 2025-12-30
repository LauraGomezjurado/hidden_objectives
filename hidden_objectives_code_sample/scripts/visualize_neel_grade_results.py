#!/usr/bin/env python3
"""Create professional visualizations for Neel-Grade experiment results."""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def load_results(results_path: Path):
    """Load experiment results."""
    with open(results_path) as f:
        return json.load(f)


def create_main_figure(results: dict, output_path: Path):
    """Create main comprehensive figure."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    analysis = results['analysis']
    probe = results['probe_handle']
    steering = results['steering_results'][0] if results['steering_results'] else None
    
    # Colors
    colors = {
        'taboo': '#2E86AB',
        'base64': '#A23B72',
        'baseline': '#6C757D',
        'steered': '#28A745',
        'transfer': '#FFC107',
    }
    
    # Panel 1: Probe Training Summary
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.5, f"Probe Accuracy:\n{probe['train_accuracy']:.1%}", 
             ha='center', va='center', fontsize=16, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax1.text(0.5, 0.2, f"Layer {probe['layer_index']}\n{probe['n_train_samples']} samples",
             ha='center', va='center', fontsize=12)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Probe Training', fontweight='bold', pad=10)
    
    # Panel 2: Baseline Metrics
    ax2 = fig.add_subplot(gs[0, 1])
    if steering:
        objectives = ['Taboo (A)', 'Base64 (B)']
        disclosure = [steering['D_A_baseline'], steering['D_B_baseline']]
        execution = [steering['E_A_baseline'], steering['E_B_baseline']]
        
        x = np.arange(len(objectives))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, disclosure, width, label='Disclosure', 
                       color=colors['taboo'], alpha=0.7)
        bars2 = ax2.bar(x + width/2, execution, width, label='Execution', 
                       color=colors['base64'], alpha=0.7)
        
        ax2.set_ylabel('Score')
        ax2.set_title('Baseline Metrics', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(objectives)
        ax2.legend()
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=9)
    
    # Panel 3: Steering Effects
    ax3 = fig.add_subplot(gs[0, 2])
    if steering:
        metrics = ['Disclosure\n(Taboo)', 'Execution\n(Taboo)', 
                  'Disclosure\n(Base64)', 'Execution\n(Base64)']
        baseline_vals = [steering['D_A_baseline'], steering['E_A_baseline'],
                        steering['D_B_baseline'], steering['E_B_baseline']]
        steered_vals = [steering['D_A_steered'], steering['E_A_steered'],
                       steering['D_B_steered'], steering['E_B_steered']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, baseline_vals, width, label='Baseline',
                       color=colors['baseline'], alpha=0.7)
        bars2 = ax3.bar(x + width/2, steered_vals, width, label=f'Steered (α={steering["alpha"]})',
                       color=colors['steered'], alpha=0.7)
        
        ax3.set_ylabel('Score')
        ax3.set_title('Steering Effects', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics, fontsize=9)
        ax3.legend()
        ax3.set_ylim(0, 1)
        ax3.grid(axis='y', alpha=0.3)
    
    # Panel 4: Transfer Effect (Main Result)
    ax4 = fig.add_subplot(gs[1, :])
    if steering:
        # Create transfer effect visualization
        transfer_effect = analysis['transfer_effect']
        exec_pres_A = analysis['execution_preservation_A']
        exec_pres_B = analysis['execution_preservation_B']
        
        # Bar chart showing key metrics
        categories = ['Transfer Effect\n(Δ Disclosure B)', 
                      'Execution Preservation A', 
                      'Execution Preservation B']
        values = [transfer_effect, exec_pres_A, exec_pres_B]
        colors_bar = [colors['transfer'], colors['taboo'], colors['base64']]
        
        bars = ax4.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax4.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Strong Transfer Threshold')
        ax4.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Weak Transfer Threshold')
        ax4.axhline(y=0.8, color='blue', linestyle='--', alpha=0.5, label='Preservation Threshold')
        
        ax4.set_ylabel('Effect Size / Preservation Ratio', fontweight='bold')
        ax4.set_title('Causal Transfer Test: Key Results', fontweight='bold', fontsize=14, pad=15)
        ax4.grid(axis='y', alpha=0.3)
        ax4.legend(loc='upper right')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            label_y = height + 0.02 if height >= 0 else height - 0.05
            ax4.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{val:.3f}',
                    ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=11, fontweight='bold')
    
    # Panel 5: Hypothesis Interpretation
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    interpretation = analysis['interpretation']
    confidence = analysis['confidence']
    detail = analysis['interpretation_detail']
    
    # Color based on hypothesis
    if interpretation == 'A':
        bg_color = '#FFE5E5'  # Light red
        text_color = '#C41E3A'
    elif interpretation == 'B':
        bg_color = '#FFF4E5'  # Light orange
        text_color = '#FF8C00'
    elif interpretation == 'C':
        bg_color = '#E5F5E5'  # Light green
        text_color = '#228B22'
    else:
        bg_color = '#F0F0F0'  # Gray
        text_color = '#333333'
    
    # Create text box
    textstr = f"Hypothesis {interpretation} ({confidence} Confidence)\n\n{detail}"
    
    ax5.text(0.5, 0.5, textstr, transform=ax5.transAxes,
            fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=1', facecolor=bg_color, 
                     edgecolor=text_color, linewidth=2, alpha=0.8),
            color=text_color, fontweight='bold')
    
    # Add failure modes if any
    if analysis.get('failure_modes'):
        failure_text = "\n\nWARNING - Failure Modes:\n" + "\n".join(f"  • {fm}" for fm in analysis['failure_modes'])
        ax5.text(0.5, 0.1, failure_text, transform=ax5.transAxes,
                fontsize=10, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3CD', 
                         edgecolor='#FFC107', linewidth=1.5, alpha=0.9),
                color='#856404')
    
    plt.suptitle('Neel-Grade Causal Transfer Experiment Results', 
                fontsize=16, fontweight='bold', y=0.98)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Main figure saved to: {output_path}")


def create_transfer_effect_figure(results: dict, output_path: Path):
    """Create focused figure on transfer effect."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    analysis = results['analysis']
    steering = results['steering_results'][0] if results['steering_results'] else None
    
    if not steering:
        return
    
    # Left panel: Before/After comparison
    categories = ['Taboo\nDisclosure', 'Taboo\nExecution', 
                 'Base64\nDisclosure', 'Base64\nExecution']
    baseline = [steering['D_A_baseline'], steering['E_A_baseline'],
               steering['D_B_baseline'], steering['E_B_baseline']]
    steered = [steering['D_A_steered'], steering['E_A_steered'],
              steering['D_B_steered'], steering['E_B_steered']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline, width, label='Baseline', 
                   color='#6C757D', alpha=0.7, edgecolor='black')
    bars2 = ax1.bar(x + width/2, steered, width, label=f'After Steering (α={steering["alpha"]})', 
                   color='#28A745', alpha=0.7, edgecolor='black')
    
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Before vs After Steering Intervention', fontweight='bold', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
    
    # Right panel: Effect sizes
    effects = ['Disclosure A\n(Source)', 'Execution A\n(Source)',
              'Disclosure B\n(Transfer)', 'Execution B\n(Transfer)']
    deltas = [steering['delta_D_A'], steering['delta_E_A'],
             steering['delta_D_B'], steering['delta_E_B']]
    
    colors_effect = ['#2E86AB', '#2E86AB', '#FFC107', '#A23B72']
    
    bars = ax2.bar(effects, deltas, color=colors_effect, alpha=0.7, 
                  edgecolor='black', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Strong Transfer')
    ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Weak Transfer')
    
    ax2.set_ylabel('Change (Δ)', fontweight='bold')
    ax2.set_title('Effect Sizes: Source vs Transfer', fontweight='bold', fontsize=13)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Adjust y-limits to give space for labels and avoid overlap
    y_min = min(deltas) - 0.1
    y_max = max(deltas) + 0.15
    ax2.set_ylim(y_min, y_max)
    
    # Add value labels with better positioning to avoid threshold lines
    for bar, val in zip(bars, deltas):
        height = bar.get_height()
        # Position labels above bars, but avoid threshold lines
        if height >= 0:
            # For positive values, place above bar
            if height < 0.1:
                label_y = height + 0.02  # Below weak transfer line
            elif height < 0.2:
                label_y = height + 0.02  # Between weak and strong
            else:
                label_y = height + 0.03  # Above strong transfer line
            va = 'bottom'
        else:
            # For negative values, place below bar
            label_y = height - 0.04
            va = 'top'
        
        ax2.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{val:+.3f}',
                ha='center', va=va, 
                fontsize=10, fontweight='bold')
    
    # Rotate x-axis labels to prevent overlap
    ax2.set_xticklabels(effects, rotation=15, ha='right')
    
    plt.suptitle('Causal Transfer Analysis: Taboo to Base64', 
                fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Transfer effect figure saved to: {output_path}")


def create_summary_figure(results: dict, output_path: Path):
    """Create executive summary figure."""
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    analysis = results['analysis']
    probe = results['probe_handle']
    steering = results['steering_results'][0] if results['steering_results'] else None
    
    # Panel 1: Hypothesis Decision
    ax1 = fig.add_subplot(gs[0, 0])
    interpretation = analysis['interpretation']
    confidence = analysis['confidence']
    
    # Create hypothesis comparison
    hypotheses = ['A\n(Independent)', 'B\n(Partially Shared)', 'C\n(Shared)']
    support_scores = [0.0, 0.0, 0.0]
    
    if interpretation == 'A':
        support_scores[0] = 1.0
        colors_hyp = ['#C41E3A', '#E0E0E0', '#E0E0E0']
    elif interpretation == 'B':
        support_scores[1] = 1.0
        colors_hyp = ['#E0E0E0', '#FF8C00', '#E0E0E0']
    elif interpretation == 'C':
        support_scores[2] = 1.0
        colors_hyp = ['#E0E0E0', '#E0E0E0', '#228B22']
    
    bars = ax1.bar(hypotheses, support_scores, color=colors_hyp, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    ax1.set_ylabel('Support Level', fontweight='bold')
    ax1.set_title(f'Hypothesis Decision: {interpretation} ({confidence})', 
                 fontweight='bold', fontsize=12)
    ax1.set_ylim(0, 1.2)
    ax1.set_yticks([0, 0.5, 1.0])
    ax1.set_yticklabels(['None', 'Partial', 'Strong'])
    ax1.grid(axis='y', alpha=0.3)
    
    # Panel 2: Key Metrics Summary
    ax2 = fig.add_subplot(gs[0, 1])
    if steering:
        metrics_data = {
            'Probe Accuracy': probe['train_accuracy'],
            'Transfer Effect': analysis['transfer_effect'],
            'Exec Preserve A': min(analysis['execution_preservation_A'], 1.0),
            'Exec Preserve B': max(analysis['execution_preservation_B'], 0.0),
        }
        
        bars = ax2.barh(list(metrics_data.keys()), list(metrics_data.values()),
                       color=['#2E86AB', '#FFC107', '#28A745', '#DC3545'],
                       alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Score / Effect Size', fontweight='bold')
        ax2.set_title('Key Metrics Summary', fontweight='bold', fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, metrics_data.values()):
            width = bar.get_width()
            ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}',
                    ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Panel 3: Transfer Effect Visualization
    ax3 = fig.add_subplot(gs[1, :])
    if steering:
        # Create a clear visualization of the transfer test
        transfer_effect = analysis['transfer_effect']
        
        # Draw arrows showing the experiment flow
        ax3.arrow(0.1, 0.5, 0.2, 0, head_width=0.05, head_length=0.02, 
                 fc='#2E86AB', ec='#2E86AB', linewidth=3, label='Train Probe')
        ax3.text(0.2, 0.65, 'Train Probe\non Taboo', ha='center', fontsize=10, fontweight='bold')
        
        ax3.arrow(0.35, 0.5, 0.15, 0, head_width=0.05, head_length=0.02, 
                 fc='#28A745', ec='#28A745', linewidth=3, label='Test on Taboo')
        ax3.text(0.425, 0.65, 'Test Steering\non Taboo', ha='center', fontsize=10, fontweight='bold')
        
        ax3.arrow(0.55, 0.5, 0.15, 0, head_width=0.05, head_length=0.02, 
                 fc='#FFC107', ec='#FFC107', linewidth=3, label='Transfer Test')
        ax3.text(0.625, 0.65, 'Apply Same Handle\nto Base64', ha='center', fontsize=10, fontweight='bold')
        
        # Show result
        result_color = '#C41E3A' if abs(transfer_effect) < 0.1 else '#FF8C00' if transfer_effect < 0.2 else '#228B22'
        result_text = 'NO TRANSFER' if abs(transfer_effect) < 0.1 else 'PARTIAL TRANSFER' if transfer_effect < 0.2 else 'STRONG TRANSFER'
        
        ax3.text(0.8, 0.5, f'Result:\n{result_text}\nΔ = {transfer_effect:.3f}', 
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=result_color, 
                         edgecolor='black', linewidth=2, alpha=0.3),
                color=result_color)
        
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Experiment Flow: Causal Transfer Test', fontweight='bold', fontsize=13, pad=20)
    
    plt.suptitle('Neel-Grade Experiment: Executive Summary', 
                fontsize=16, fontweight='bold', y=0.98)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Summary figure saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Neel-Grade experiment results')
    parser.add_argument('--results', type=str, 
                       default='outputs/neel_grade_quick/neel_grade_results.json',
                       help='Path to results JSON file')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/neel_grade_quick',
                       help='Output directory for figures')
    args = parser.parse_args()
    
    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        return
    
    print(f"Loading results from {results_path}...")
    results = load_results(results_path)
    
    print("Creating visualizations...")
    
    # Create main comprehensive figure
    create_main_figure(results, output_dir / 'neel_grade_main.png')
    
    # Create transfer effect focused figure
    create_transfer_effect_figure(results, output_dir / 'neel_grade_transfer.png')
    
    # Create executive summary
    create_summary_figure(results, output_dir / 'neel_grade_summary.png')
    
    print(f"\nAll figures saved to: {output_dir}")
    print("Files created:")
    print("  - neel_grade_main.png (comprehensive overview)")
    print("  - neel_grade_transfer.png (transfer effect focus)")
    print("  - neel_grade_summary.png (executive summary)")


if __name__ == '__main__':
    main()

