#!/usr/bin/env python3
"""
Task B3: Simulation Parameter Variation (10 Points)

Test 5 scenarios:
1. Baseline
2. Large curve radius
3. High sensor noise
4. Small observation radius
5. Large observation radius

Analyze impact of curve radius, sensor noise, and observation radius
Create table and plots comparing performance across scenarios
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from run_simulation import run_simulation


def task_b3_simulation_parameter_variation(optimal_process_noise_xy=1e-4,
                                           optimal_process_noise_theta=1e-4,
                                           optimal_measurement_noise=0.01):
    """
    Experiment with varying simulation parameters
    
    Args:
        optimal_process_noise_xy: Optimal process noise from B1
        optimal_process_noise_theta: Optimal process noise theta from B1
        optimal_measurement_noise: Optimal measurement noise from B2
    """
    print("=" * 80)
    print("TASK B3: Simulation Parameter Variation Experiment")
    print("=" * 80)
    
    # Base parameters (optimal from B1 and B2)
    duration = 50.0
    dt = 0.1
    process_noise_xy = optimal_process_noise_xy
    process_noise_theta = optimal_process_noise_theta
    measurement_noise_xy = optimal_measurement_noise
    
    # Define 5 scenarios
    scenarios = [
        {
            'name': '1. Baseline',
            'description': 'Standard parameters, moderate noise, medium observation radius',
            'params': {
                'curve_radius': 5.0,
                'odometry_noise_xy': 0.1,
                'odometry_noise_theta': 0.05,
                'observation_noise': 0.1,
                'observation_radius': 5.0
            }
        },
        {
            'name': '2. Large Curve Radius',
            'description': 'Larger circular trajectory, slower angular velocity',
            'params': {
                'curve_radius': 10.0,  # Larger radius
                'odometry_noise_xy': 0.1,
                'odometry_noise_theta': 0.05,
                'observation_noise': 0.1,
                'observation_radius': 5.0
            }
        },
        {
            'name': '3. High Sensor Noise',
            'description': 'Increased odometry and observation noise',
            'params': {
                'curve_radius': 5.0,
                'odometry_noise_xy': 0.3,  # 3x higher
                'odometry_noise_theta': 0.15,  # 3x higher
                'observation_noise': 0.3,  # 3x higher
                'observation_radius': 5.0
            }
        },
        {
            'name': '4. Small Observation Radius',
            'description': 'Limited landmark visibility',
            'params': {
                'curve_radius': 5.0,
                'odometry_noise_xy': 0.1,
                'odometry_noise_theta': 0.05,
                'observation_noise': 0.1,
                'observation_radius': 2.5  # Half of baseline
            }
        },
        {
            'name': '5. Large Observation Radius',
            'description': 'Extended landmark visibility',
            'params': {
                'curve_radius': 5.0,
                'odometry_noise_xy': 0.1,
                'odometry_noise_theta': 0.05,
                'observation_noise': 0.1,
                'observation_radius': 10.0  # Double baseline
            }
        }
    ]
    
    results_list = []
    
    print(f"\nFixed UKF Parameters (Optimal from B1 and B2):")
    print(f"  Process Noise (XY): {process_noise_xy}")
    print(f"  Process Noise (Theta): {process_noise_theta}")
    print(f"  Measurement Noise: {measurement_noise_xy}")
    print(f"  Duration: {duration}s")
    print(f"\nRunning {len(scenarios)} scenarios...\n")
    
    # Run simulations for each scenario
    for i, scenario in enumerate(scenarios):
        print(f"\n{'='*70}")
        print(f"Scenario {i+1}/{len(scenarios)}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Parameters: {scenario['params']}")
        print(f"{'='*70}")
        
        results = run_simulation(
            duration=duration,
            dt=dt,
            process_noise_xy=process_noise_xy,
            process_noise_theta=process_noise_theta,
            measurement_noise_xy=measurement_noise_xy,
            **scenario['params'],
            visualize=False,
            save_plot=None
        )
        
        # Calculate metrics
        errors = np.array(results['errors'])
        
        # Convergence time (when error drops below threshold)
        threshold = 0.1  # 10cm
        convergence_idx = np.where(errors < threshold)[0]
        convergence_time = results['times'][convergence_idx[0]] if len(convergence_idx) > 0 else duration
        
        # Stability after convergence
        if len(convergence_idx) > 0:
            stable_errors = errors[convergence_idx[0]:]
            stability = np.std(stable_errors)
        else:
            stability = np.std(errors)
        
        # Tracking accuracy (mean of last 25%)
        last_quarter_idx = int(len(errors) * 0.75)
        tracking_accuracy = np.mean(errors[last_quarter_idx:])
        
        # Peak error (95th percentile to avoid outliers)
        peak_error = np.percentile(errors, 95)
        
        results_list.append({
            'scenario': scenario['name'],
            'description': scenario['description'],
            'rmse': results['rmse'],
            'mean_error': results['mean_error'],
            'max_error': results['max_error'],
            'peak_error': peak_error,
            'convergence_time': convergence_time,
            'stability': stability,
            'tracking_accuracy': tracking_accuracy,
            'errors': errors,
            'times': results['times'],
            'params': scenario['params']
        })
        
        print(f"\nResults:")
        print(f"  RMSE: {results['rmse']:.4f} m")
        print(f"  Mean Error: {results['mean_error']:.4f} m")
        print(f"  Peak Error (95%): {peak_error:.4f} m")
        print(f"  Convergence Time: {convergence_time:.2f} s")
        print(f"  Stability: {stability:.4f} m")
        print(f"  Tracking Accuracy: {tracking_accuracy:.4f} m")
    
    # Generate comprehensive report
    generate_b3_report(results_list)
    
    # Create visualizations
    create_b3_plots(results_list)
    
    return results_list


def generate_b3_report(results_list):
    """Generate comprehensive report for Task B3"""
    
    print("\n" + "=" * 80)
    print("TASK B3: Summary Report")
    print("=" * 80)
    
    # Create comparison table
    print("\nPerformance Comparison Across Scenarios:")
    print("-" * 100)
    print(f"{'Scenario':<25} {'RMSE':<10} {'Peak Err':<10} {'Conv. Time':<12} "
          f"{'Stability':<12} {'Tracking':<12}")
    print(f"{'':25} {'(m)':<10} {'(m)':<10} {'(s)':<12} {'(m)':<12} {'(m)':<12}")
    print("-" * 100)
    
    for r in results_list:
        print(f"{r['scenario']:<25} {r['rmse']:<10.4f} {r['peak_error']:<10.4f} "
              f"{r['convergence_time']:<12.2f} {r['stability']:<12.4f} "
              f"{r['tracking_accuracy']:<12.4f}")
    
    print("\n" + "=" * 80)
    print("Detailed Analysis:")
    print("=" * 80)
    
    # Scenario 1 vs 2: Impact of Curve Radius
    print("\n1. Impact of Curve Radius (Baseline vs Large Radius):")
    print("-" * 80)
    baseline = results_list[0]
    large_radius = results_list[1]
    
    print(f"Baseline (R=5.0m):      RMSE={baseline['rmse']:.4f}m, "
          f"Conv={baseline['convergence_time']:.2f}s")
    print(f"Large Radius (R=10.0m): RMSE={large_radius['rmse']:.4f}m, "
          f"Conv={large_radius['convergence_time']:.2f}s")
    
    rmse_diff = ((large_radius['rmse'] - baseline['rmse']) / baseline['rmse']) * 100
    conv_diff = ((large_radius['convergence_time'] - baseline['convergence_time']) / 
                 baseline['convergence_time']) * 100
    
    print(f"\nChange: RMSE {rmse_diff:+.1f}%, Convergence Time {conv_diff:+.1f}%")
    print("\nObservation:")
    print("  - Larger radius means slower angular velocity")
    print("  - Less dynamic motion → easier to track")
    print("  - Smoother trajectory → potentially better performance")
    
    # Scenario 1 vs 3: Impact of Sensor Noise
    print("\n2. Impact of High Sensor Noise (Baseline vs High Noise):")
    print("-" * 80)
    high_noise = results_list[2]
    
    print(f"Baseline (σ=0.1m):    RMSE={baseline['rmse']:.4f}m, "
          f"Stability={baseline['stability']:.4f}m")
    print(f"High Noise (σ=0.3m):  RMSE={high_noise['rmse']:.4f}m, "
          f"Stability={high_noise['stability']:.4f}m")
    
    rmse_diff = ((high_noise['rmse'] - baseline['rmse']) / baseline['rmse']) * 100
    stab_diff = ((high_noise['stability'] - baseline['stability']) / baseline['stability']) * 100
    
    print(f"\nChange: RMSE {rmse_diff:+.1f}%, Stability {stab_diff:+.1f}%")
    print("\nObservation:")
    print("  - 3x increase in sensor noise significantly degrades performance")
    print("  - UKF still converges but with higher error")
    print("  - Demonstrates importance of sensor quality")
    
    # Scenario 4 vs 5: Impact of Observation Radius
    print("\n3. Impact of Observation Radius (Small vs Large):")
    print("-" * 80)
    small_radius = results_list[3]
    large_obs_radius = results_list[4]
    
    print(f"Small Obs. Radius (2.5m):  RMSE={small_radius['rmse']:.4f}m, "
          f"Conv={small_radius['convergence_time']:.2f}s")
    print(f"Baseline (5.0m):           RMSE={baseline['rmse']:.4f}m, "
          f"Conv={baseline['convergence_time']:.2f}s")
    print(f"Large Obs. Radius (10.0m): RMSE={large_obs_radius['rmse']:.4f}m, "
          f"Conv={large_obs_radius['convergence_time']:.2f}s")
    
    print("\nObservation:")
    print("  - Larger observation radius → more landmark observations")
    print("  - More observations → better localization accuracy")
    print("  - Faster convergence with more information")
    print("  - Diminishing returns beyond certain radius")
    
    # Overall conclusions
    print("\n" + "=" * 80)
    print("Key Findings:")
    print("=" * 80)
    print("\n1. Trajectory Complexity:")
    print("   - Simpler trajectories (larger radius) are easier to track")
    print("   - High angular velocity increases prediction uncertainty")
    
    print("\n2. Sensor Noise Impact:")
    print("   - Sensor quality directly affects filter performance")
    print("   - UKF is robust to moderate noise")
    print("   - Tuning process/measurement noise is critical")
    
    print("\n3. Observation Availability:")
    print("   - More landmarks in view → better localization")
    print("   - Observation radius significantly affects convergence")
    print("   - Trade-off: computational cost vs. accuracy")
    
    print("\n4. Best Performance:")
    best_idx = np.argmin([r['rmse'] for r in results_list])
    best = results_list[best_idx]
    print(f"   Scenario: {best['scenario']}")
    print(f"   RMSE: {best['rmse']:.4f} m")
    print(f"   Description: {best['description']}")


def create_b3_plots(results_list):
    """Create visualization plots for Task B3"""
    
    fig = plt.figure(figsize=(18, 12))
    
    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot 1: RMSE Comparison
    ax1 = plt.subplot(3, 3, 1)
    scenarios = [r['scenario'] for r in results_list]
    rmses = [r['rmse'] for r in results_list]
    bars1 = ax1.bar(range(len(scenarios)), rmses, color=colors)
    ax1.set_xticks(range(len(scenarios)))
    ax1.set_xticklabels([s.split('.')[0] for s in scenarios], fontsize=10)
    ax1.set_ylabel('RMSE (m)', fontsize=11)
    ax1.set_title('Root Mean Square Error', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Convergence Time Comparison
    ax2 = plt.subplot(3, 3, 2)
    conv_times = [r['convergence_time'] for r in results_list]
    bars2 = ax2.bar(range(len(scenarios)), conv_times, color=colors)
    ax2.set_xticks(range(len(scenarios)))
    ax2.set_xticklabels([s.split('.')[0] for s in scenarios], fontsize=10)
    ax2.set_ylabel('Convergence Time (s)', fontsize=11)
    ax2.set_title('Convergence Speed', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Stability Comparison
    ax3 = plt.subplot(3, 3, 3)
    stabilities = [r['stability'] for r in results_list]
    bars3 = ax3.bar(range(len(scenarios)), stabilities, color=colors)
    ax3.set_xticks(range(len(scenarios)))
    ax3.set_xticklabels([s.split('.')[0] for s in scenarios], fontsize=10)
    ax3.set_ylabel('Stability (std, m)', fontsize=11)
    ax3.set_title('Error Stability', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Tracking Accuracy Comparison
    ax4 = plt.subplot(3, 3, 4)
    tracking = [r['tracking_accuracy'] for r in results_list]
    bars4 = ax4.bar(range(len(scenarios)), tracking, color=colors)
    ax4.set_xticks(range(len(scenarios)))
    ax4.set_xticklabels([s.split('.')[0] for s in scenarios], fontsize=10)
    ax4.set_ylabel('Tracking Accuracy (m)', fontsize=11)
    ax4.set_title('Steady-State Tracking', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Peak Error Comparison
    ax5 = plt.subplot(3, 3, 5)
    peak_errors = [r['peak_error'] for r in results_list]
    bars5 = ax5.bar(range(len(scenarios)), peak_errors, color=colors)
    ax5.set_xticks(range(len(scenarios)))
    ax5.set_xticklabels([s.split('.')[0] for s in scenarios], fontsize=10)
    ax5.set_ylabel('Peak Error (95%, m)', fontsize=11)
    ax5.set_title('Peak Error (95th Percentile)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Radar chart for multi-metric comparison
    ax6 = plt.subplot(3, 3, 6, projection='polar')
    
    # Normalize metrics for radar chart
    categories = ['RMSE', 'Conv. Time', 'Stability', 'Tracking', 'Peak Error']
    
    # Use baseline (scenario 1) for normalization
    baseline_values = [rmses[0], conv_times[0], stabilities[0], tracking[0], peak_errors[0]]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for i, r in enumerate(results_list):
        values = [
            rmses[i] / baseline_values[0],
            conv_times[i] / baseline_values[1],
            stabilities[i] / baseline_values[2],
            tracking[i] / baseline_values[3],
            peak_errors[i] / baseline_values[4]
        ]
        values += values[:1]
        ax6.plot(angles, values, 'o-', linewidth=2, label=r['scenario'].split('.')[0], color=colors[i])
        ax6.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories, fontsize=9)
    ax6.set_ylim(0, 2)
    ax6.set_title('Relative Performance\n(normalized to baseline)', 
                 fontsize=12, fontweight='bold', pad=20)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax6.grid(True)
    
    # Plot 7-8-9: Error trajectories
    ax7 = plt.subplot(3, 1, 3)
    for i, r in enumerate(results_list):
        ax7.plot(r['times'], r['errors'], linewidth=1.5, 
                label=r['scenario'], color=colors[i], alpha=0.8)
    
    ax7.axhline(y=0.1, color='gray', linestyle=':', linewidth=1, label='Threshold (10cm)')
    ax7.set_xlabel('Time (s)', fontsize=12)
    ax7.set_ylabel('Position Error (m)', fontsize=12)
    ax7.set_title('Error Evolution Across All Scenarios', fontsize=13, fontweight='bold')
    ax7.legend(loc='upper right', fontsize=10)
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'task_b3_simulation_parameter_variation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {output_path}")
    
    # Also save results table as CSV
    csv_data = []
    for r in results_list:
        csv_data.append({
            'Scenario': r['scenario'],
            'Description': r['description'],
            'RMSE (m)': r['rmse'],
            'Mean Error (m)': r['mean_error'],
            'Peak Error (m)': r['peak_error'],
            'Max Error (m)': r['max_error'],
            'Convergence Time (s)': r['convergence_time'],
            'Stability (m)': r['stability'],
            'Tracking Accuracy (m)': r['tracking_accuracy']
        })
    
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, 'task_b3_results_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results table saved to: {csv_path}")
    
    plt.show()


if __name__ == "__main__":
    # Use optimal parameters from B1 and B2
    results = task_b3_simulation_parameter_variation(
        optimal_process_noise_xy=1e-4,
        optimal_process_noise_theta=1e-4,
        optimal_measurement_noise=0.01
    )
    print("\nTask B3 complete!")

