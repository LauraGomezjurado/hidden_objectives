#!/usr/bin/env python3
"""Analyze Experiment 3 results in the context of hypotheses A, B, C."""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_results():
    """Load all Experiment 3 results."""
    base_dir = Path("outputs/experiment_3")
    
    with open(base_dir / "steering_results.json") as f:
        steering = json.load(f)
    
    with open(base_dir / "analysis.json") as f:
        analysis = json.load(f)
    
    with open(base_dir / "directions.json") as f:
        directions = json.load(f)
    
    return steering, analysis, directions

def analyze_steering_effects(steering):
    """Analyze steering effects on both objectives."""
    print("\n" + "=" * 70)
    print("STEERING EFFECTS ANALYSIS")
    print("=" * 70)
    
    # Group by gamma
    by_gamma = {}
    for r in steering:
        gamma = r['gamma']
        if gamma not in by_gamma:
            by_gamma[gamma] = []
        by_gamma[gamma].append(r)
    
    baseline = steering[0]  # All should have same baseline
    
    print(f"\nBaseline (no steering):")
    print(f"  E_A = {baseline['baseline_E_A']:.3f}, D_A = {baseline['baseline_D_A']:.3f}")
    print(f"  E_B = {baseline['baseline_E_B']:.3f}, D_B = {baseline['baseline_D_B']:.3f}")
    
    print(f"\nSteering Effects:")
    print(f"{'γ':>6} | {'ΔE_A':>8} | {'ΔE_B':>8} | {'ΔD_A':>8} | {'ΔD_B':>8} | Interpretation")
    print("-" * 70)
    
    for gamma in sorted(by_gamma.keys()):
        r = by_gamma[gamma][0]  # Should only be one per gamma
        delta_E_A = r['E_A'] - r['baseline_E_A']
        delta_E_B = r['E_B'] - r['baseline_E_B']
        delta_D_A = r['D_A'] - r['baseline_D_A']
        delta_D_B = r['D_B'] - r['baseline_D_B']
        
        # Interpretation
        if gamma < 0:
            direction = "toward conceal"
        elif gamma > 0:
            direction = "toward reveal"
        else:
            direction = "baseline"
        
        print(f"{gamma:6.1f} | {delta_E_A:8.3f} | {delta_E_B:8.3f} | {delta_D_A:8.3f} | {delta_D_B:8.3f} | {direction}")
    
    return by_gamma, baseline

def analyze_cross_objective_transfer(analysis):
    """Analyze cross-objective transfer evidence."""
    print("\n" + "=" * 70)
    print("CROSS-OBJECTIVE TRANSFER ANALYSIS")
    print("=" * 70)
    
    if 'best_transfer_layer' in analysis:
        print(f"\nBest transfer layer: {analysis['best_transfer_layer']}")
        print(f"Transfer accuracy: {analysis.get('best_transfer_accuracy', 0):.3f}")
        print(f"\nInterpretation: {analysis.get('interpretation', 'N/A')}")
        
        # Check transfer strength
        transfer_acc = analysis.get('best_transfer_accuracy', 0)
        if transfer_acc > 0.7:
            strength = "STRONG"
        elif transfer_acc > 0.6:
            strength = "MODERATE"
        elif transfer_acc > 0.5:
            strength = "WEAK"
        else:
            strength = "NONE"
        
        print(f"Transfer strength: {strength} ({transfer_acc:.1%})")
    
    return analysis

def hypothesis_evaluation(steering, analysis, baseline):
    """Evaluate evidence for each hypothesis."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS EVALUATION")
    print("=" * 70)
    
    # Extract key metrics
    transfer_acc = analysis.get('best_transfer_accuracy', 0)
    
    # Calculate steering effects
    steering_effects = {}
    for r in steering:
        gamma = r['gamma']
        if gamma != 0:  # Skip baseline
            steering_effects[gamma] = {
                'delta_D_A': r['D_A'] - r['baseline_D_A'],
                'delta_D_B': r['D_B'] - r['baseline_D_B'],
                'delta_E_A': r['E_A'] - r['baseline_E_A'],
                'delta_E_B': r['E_B'] - r['baseline_E_B'],
            }
    
    print("\n1. HYPOTHESIS A: Independent Concealment")
    print("-" * 70)
    print("Prediction: Interventions should NOT transfer across objectives")
    print(f"Evidence:")
    print(f"  - Cross-objective transfer accuracy: {transfer_acc:.3f}")
    
    if transfer_acc < 0.55:
        print(f"  → Supports H_A: Low transfer ({transfer_acc:.1%}) suggests independence")
        h_a_support = "STRONG"
    elif transfer_acc < 0.65:
        print(f"  → Weakly supports H_A: Moderate transfer ({transfer_acc:.1%})")
        h_a_support = "WEAK"
    else:
        print(f"  → Contradicts H_A: High transfer ({transfer_acc:.1%}) suggests sharing")
        h_a_support = "CONTRADICTS"
    
    # Check if steering affects both objectives similarly
    if steering_effects:
        gamma_neg = steering_effects.get(-1.0, {})
        gamma_pos = steering_effects.get(1.0, {})
        
        if gamma_neg:
            d_a_effect_neg = abs(gamma_neg.get('delta_D_A', 0))
            d_b_effect_neg = abs(gamma_neg.get('delta_D_B', 0))
            print(f"  - Steering effect on D_A (γ=-1): {d_a_effect_neg:.3f}")
            print(f"  - Steering effect on D_B (γ=-1): {d_b_effect_neg:.3f}")
            
            if d_a_effect_neg > 0.01 and d_b_effect_neg > 0.01:
                print(f"  → Contradicts H_A: Steering affects BOTH objectives")
                h_a_support = "CONTRADICTS"
            elif d_a_effect_neg > 0.01 and d_b_effect_neg < 0.01:
                print(f"  → Supports H_A: Steering only affects Taboo")
    
    print(f"\nOverall H_A support: {h_a_support}")
    
    print("\n2. HYPOTHESIS B: Partially Shared Concealment")
    print("-" * 70)
    print("Prediction: Partial transfer, interventions weaken but don't fully break concealment")
    print(f"Evidence:")
    print(f"  - Transfer accuracy: {transfer_acc:.3f}")
    
    if 0.55 <= transfer_acc <= 0.75:
        print(f"  → Supports H_B: Moderate transfer ({transfer_acc:.1%}) suggests partial sharing")
        h_b_support = "STRONG"
    elif 0.5 <= transfer_acc < 0.55 or 0.75 < transfer_acc <= 0.85:
        print(f"  → Weakly supports H_B: Transfer ({transfer_acc:.1%}) is borderline")
        h_b_support = "MODERATE"
    else:
        print(f"  → Contradicts H_B: Transfer ({transfer_acc:.1%}) is too low/high")
        h_b_support = "WEAK"
    
    # Check steering effects
    if steering_effects:
        gamma_neg = steering_effects.get(-1.0, {})
        if gamma_neg:
            d_a_effect = abs(gamma_neg.get('delta_D_A', 0))
            d_b_effect = abs(gamma_neg.get('delta_D_B', 0))
            if 0.01 < d_a_effect < 0.1 and 0.01 < d_b_effect < 0.1:
                print(f"  - Partial effects on both objectives supports H_B")
                h_b_support = "STRONG"
    
    print(f"\nOverall H_B support: {h_b_support}")
    
    print("\n3. HYPOTHESIS C: Shared Concealment")
    print("-" * 70)
    print("Prediction: Strong transfer, single intervention disrupts both objectives")
    print(f"Evidence:")
    print(f"  - Transfer accuracy: {transfer_acc:.3f}")
    
    if transfer_acc > 0.75:
        print(f"  → Supports H_C: Strong transfer ({transfer_acc:.1%}) suggests shared mechanism")
        h_c_support = "STRONG"
    elif transfer_acc > 0.65:
        print(f"  → Moderately supports H_C: Good transfer ({transfer_acc:.1%})")
        h_c_support = "MODERATE"
    else:
        print(f"  → Contradicts H_C: Weak transfer ({transfer_acc:.1%}) suggests separation")
        h_c_support = "WEAK"
    
    # Check if steering strongly affects both
    if steering_effects:
        gamma_neg = steering_effects.get(-1.0, {})
        gamma_pos = steering_effects.get(1.0, {})
        
        if gamma_neg:
            d_a_effect = abs(gamma_neg.get('delta_D_A', 0))
            d_b_effect = abs(gamma_neg.get('delta_D_B', 0))
            if d_a_effect > 0.05 and d_b_effect > 0.05:
                print(f"  - Strong steering effects on BOTH objectives supports H_C")
                h_c_support = "STRONG"
            elif d_a_effect > 0.01 and d_b_effect > 0.01:
                print(f"  - Steering affects both objectives (moderate) supports H_C")
                if h_c_support == "WEAK":
                    h_c_support = "MODERATE"
    
    print(f"\nOverall H_C support: {h_c_support}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nHypothesis A (Independent):     {h_a_support}")
    print(f"Hypothesis B (Partially Shared): {h_b_support}")
    print(f"Hypothesis C (Shared):           {h_c_support}")
    
    # Determine most supported
    supports = {
        'A': h_a_support,
        'B': h_b_support,
        'C': h_c_support,
    }
    
    # Rank by strength
    strength_order = {'STRONG': 3, 'MODERATE': 2, 'WEAK': 1, 'CONTRADICTS': 0}
    ranked = sorted(supports.items(), key=lambda x: strength_order.get(x[1], 0), reverse=True)
    
    print(f"\nMost supported hypothesis: H_{ranked[0][0]} ({ranked[0][1]})")
    
    return {
        'H_A': h_a_support,
        'H_B': h_b_support,
        'H_C': h_c_support,
        'transfer_accuracy': transfer_acc,
    }

def create_visualization(steering, analysis, output_dir):
    """Create visualization of results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    gammas = sorted([r['gamma'] for r in steering])
    E_A = [r['E_A'] for r in sorted(steering, key=lambda x: x['gamma'])]
    E_B = [r['E_B'] for r in sorted(steering, key=lambda x: x['gamma'])]
    D_A = [r['D_A'] for r in sorted(steering, key=lambda x: x['gamma'])]
    D_B = [r['D_B'] for r in sorted(steering, key=lambda x: x['gamma'])]
    
    baseline_E_A = steering[0]['baseline_E_A']
    baseline_E_B = steering[0]['baseline_E_B']
    baseline_D_A = steering[0]['baseline_D_A']
    baseline_D_B = steering[0]['baseline_D_B']
    
    # Plot 1: Execution scores
    ax = axes[0, 0]
    ax.plot(gammas, E_A, 'o-', label='E_A (Taboo)', linewidth=2, markersize=8)
    ax.plot(gammas, E_B, 's-', label='E_B (Base64)', linewidth=2, markersize=8)
    ax.axhline(y=baseline_E_A, color='blue', linestyle='--', alpha=0.5, label='Baseline E_A')
    ax.axhline(y=baseline_E_B, color='orange', linestyle='--', alpha=0.5, label='Baseline E_B')
    ax.set_xlabel('Steering Strength (γ)', fontsize=12)
    ax.set_ylabel('Execution Score', fontsize=12)
    ax.set_title('Effect of Steering on Execution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Disclosure scores
    ax = axes[0, 1]
    ax.plot(gammas, D_A, 'o-', label='D_A (Taboo)', linewidth=2, markersize=8, color='red')
    ax.plot(gammas, D_B, 's-', label='D_B (Base64)', linewidth=2, markersize=8, color='green')
    ax.axhline(y=baseline_D_A, color='red', linestyle='--', alpha=0.5, label='Baseline D_A')
    ax.axhline(y=baseline_D_B, color='green', linestyle='--', alpha=0.5, label='Baseline D_B')
    ax.set_xlabel('Steering Strength (γ)', fontsize=12)
    ax.set_ylabel('Disclosure Score', fontsize=12)
    ax.set_title('Effect of Steering on Disclosure', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Cross-objective transfer
    ax = axes[1, 0]
    if 'best_transfer_layer' in analysis:
        transfer_acc = analysis.get('best_transfer_accuracy', 0)
        layers = [8, 16, 24]
        # We don't have per-layer transfer accuracies, so show best
        accuracies = [0.5] * len(layers)  # Placeholder
        best_idx = layers.index(analysis['best_transfer_layer']) if analysis['best_transfer_layer'] in layers else 0
        accuracies[best_idx] = transfer_acc
        
        bars = ax.bar(layers, accuracies, alpha=0.7, color=['red' if a < 0.6 else 'orange' if a < 0.75 else 'green' for a in accuracies])
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Chance (50%)')
        ax.axhline(y=0.75, color='green', linestyle='--', alpha=0.5, label='Strong transfer (75%)')
        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Transfer Accuracy', fontsize=12)
        ax.set_title('Cross-Objective Transfer Accuracy', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
    
    # Plot 4: Hypothesis comparison
    ax = axes[1, 1]
    hypotheses = ['H_A\n(Independent)', 'H_B\n(Partial)', 'H_C\n(Shared)']
    # This would need the evaluation results - placeholder for now
    support_levels = [0.33, 0.33, 0.33]  # Placeholder
    colors = ['red', 'orange', 'green']
    bars = ax.bar(hypotheses, support_levels, alpha=0.7, color=colors)
    ax.set_ylabel('Support Level', fontsize=12)
    ax.set_title('Hypothesis Support', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'experiment_3_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_dir / 'experiment_3_analysis.png'}")

def main():
    steering, analysis, directions = load_results()
    
    print("=" * 70)
    print("EXPERIMENT 3: CONCEALMENT DIRECTION ANALYSIS")
    print("=" * 70)
    
    by_gamma, baseline = analyze_steering_effects(steering)
    transfer_analysis = analyze_cross_objective_transfer(analysis)
    hypothesis_results = hypothesis_evaluation(steering, analysis, baseline)
    
    create_visualization(steering, analysis, "outputs/experiment_3")
    
    # Save summary
    summary = {
        'hypothesis_support': {
            'H_A': hypothesis_results['H_A'],
            'H_B': hypothesis_results['H_B'],
            'H_C': hypothesis_results['H_C'],
        },
        'transfer_accuracy': hypothesis_results['transfer_accuracy'],
        'steering_effects': {
            str(gamma): {
                'delta_D_A': r['D_A'] - r['baseline_D_A'],
                'delta_D_B': r['D_B'] - r['baseline_D_B'],
            }
            for gamma, r_list in by_gamma.items()
            for r in r_list
        },
    }
    
    with open("outputs/experiment_3/hypothesis_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Saved hypothesis analysis to: outputs/experiment_3/hypothesis_analysis.json")

if __name__ == "__main__":
    main()


