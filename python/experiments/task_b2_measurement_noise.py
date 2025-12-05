#!/usr/bin/env python3
"""
Task B2: Measurement Noise Variation (10 Points)

Run 5 simulations varying measurement noise using optimal process noise from B1:
- Measure convergence speed, stability, and tracking accuracy
- Plot convergence vs. measurement noise
- Document optimal parameters and compare with process noise effects
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from run_simulation import run_simulation


def task_b2_measurement_noise_variation(optimal_process_noise_xy=1e-4, 
                                        optimal_process_noise_theta=1e-4):
    """
    Experiment with varying measurement noise while keeping process noise constant
    
    Args:
        optimal_process_noise_xy: Optimal process noise from Task B1
        optimal_process_noise_theta: Optimal process noise for theta from Task B1
    """
    print("=" * 80)
    print("TASK B2: Measurement Noise Variation Experiment")
    print("=" * 80)
    
    # Fixed parameters (using optimal from B1)
    duration = 40.0
    dt = 0.1
    process_noise_xy = optimal_process_noise_xy
    process_noise_theta = optimal_process_noise_theta
    odometry_noise_xy = 0.1
    odometry_noise_theta = 0.05
    observation_noise = 0.1
    observation_radius = 5.0
    
    # Varying measurement noise (5 different values)
    measurement_noise_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    results_list = []
    
    print(f"\nFixed Parameters (Optimal from B1):")
    print(f"  Process Noise (XY): {process_noise_xy}")
    print(f"  Process Noise (Theta): {process_noise_theta}")
    print(f"  Duration: {duration}s")
    print(f"  Observation Radius: {observation_radius}m")
    print(f"\nRunning {len(measurement_noise_values)} simulations...\n")
    
    # Run simulations with different measurement noise values
    for i, mn in enumerate(measurement_noise_values):
        print(f"\n{'='*70}")
        print(f"Simulation {i+1}/{len(measurement_noise_values)}")
        print(f"Measurement Noise: {mn}")
        print(f"{'='*70}")
        
        results = run_simulation(
            duration=duration,
            dt=dt,
            process_noise_xy=process_noise_xy,
            process_noise_theta=process_noise_theta,
            measurement_noise_xy=mn,
            odometry_noise_xy=odometry_noise_xy,
            odometry_noise_theta=odometry_noise_theta,
            observation_noise=observation_noise,
            observation_radius=observation_radius,
            visualize=False,
            save_plot=None
        )
        
        # Calculate convergence time (when error drops below threshold)
        errors = np.array(results['errors'])
        threshold = 0.1  # 10cm threshold
        convergence_idx = np.where(errors < threshold)[0]
        convergence_time = results['times'][convergence_idx[0]] if len(convergence_idx) > 0 else duration
        
        # Calculate stability (variance of error after convergence)
        if len(convergence_idx) > 0:
            stable_errors = errors[convergence_idx[0]:]
            stability = np.std(stable_errors)
        else:
            stability = np.std(errors)
        
        # Calculate tracking accuracy (mean of last 25% of trajectory)
        last_quarter_idx = int(len(errors) * 0.75)
        tracking_accuracy = np.mean(errors[last_quarter_idx:])
        
        results_list.append({
            'measurement_noise': mn,
            'rmse': results['rmse'],
            'mean_error': results['mean_error'],
            'max_error': results['max_error'],
            'convergence_time': convergence_time,
            'stability': stability,
            'tracking_accuracy': tracking_accuracy,
            'errors': errors,
            'times': results['times']
        })
        
        print(f"\nResults:")
        print(f"  RMSE: {results['rmse']:.4f} m")
        print(f"  Convergence Time: {convergence_time:.2f} s")
        print(f"  Stability (std): {stability:.4f} m")
        print(f"  Tracking Accuracy: {tracking_accuracy:.4f} m")
    
    # Generate comprehensive report
    print("\n" + "=" * 80)
    print("TASK B2: Summary Report")
    print("=" * 80)
    print("\nMeasurement Noise Variation Results:")
    print("-" * 80)
    print(f"{'Meas. Noise':<15} {'RMSE (m)':<12} {'Conv. Time (s)':<15} "
          f"{'Stability (m)':<15} {'Tracking (m)':<15}")
    print("-" * 80)
    
    for r in results_list:
        print(f"{r['measurement_noise']:<15.3f} {r['rmse']:<12.4f} "
              f"{r['convergence_time']:<15.2f} {r['stability']:<15.4f} "
              f"{r['tracking_accuracy']:<15.4f}")
    
    # Find optimal parameters
    min_rmse_idx = np.argmin([r['rmse'] for r in results_list])
    optimal = results_list[min_rmse_idx]
    
    print("\n" + "-" * 80)
    print("Optimal Parameters:")
    print(f"  Measurement Noise: {optimal['measurement_noise']:.3f}")
    print(f"  RMSE: {optimal['rmse']:.4f} m")
    print(f"  Convergence Time: {optimal['convergence_time']:.2f} s")
    print(f"  Stability: {optimal['stability']:.4f} m")
    print(f"  Tracking Accuracy: {optimal['tracking_accuracy']:.4f} m")
    
    print("\nBehavior Analysis:")
    print("-" * 80)
    print("1. Low Measurement Noise (0.001 to 0.005):")
    print("   - Filter trusts measurements highly")
    print("   - Faster convergence due to confident measurements")
    print("   - Very good tracking accuracy")
    print("   - May be overly optimistic if actual noise is higher")
    
    print("\n2. Moderate Measurement Noise (0.01 to 0.05):")
    print("   - Balanced trust, realistic for most sensors")
    print("   - Good convergence and tracking performance")
    print("   - Robust to moderate measurement errors")
    
    print("\n3. High Measurement Noise (0.1):")
    print("   - Filter discounts measurements, relies more on prediction")
    print("   - Slower convergence, less responsive to observations")
    print("   - More stable but potentially less accurate")
    
    print("\nComparison with Process Noise Effects:")
    print("-" * 80)
    print("Process Noise (Task B1):")
    print("  - Affects prediction uncertainty")
    print("  - Controls how much model adapts over time")
    print("  - Higher values = less trust in motion model")
    
    print("\nMeasurement Noise (Task B2):")
    print("  - Affects measurement update uncertainty")
    print("  - Controls how much filter trusts observations")
    print("  - Higher values = less trust in landmark observations")
    
    print("\nKey Insight:")
    print("  - Process and measurement noise must be balanced")
    print("  - Their ratio determines prediction vs measurement weighting")
    print("  - Optimal values depend on actual sensor characteristics")
    
    # Create visualizations
    create_b2_plots(results_list)
    
    return results_list


def create_b2_plots(results_list):
    """Create visualization plots for Task B2"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: RMSE vs Measurement Noise
    ax1 = plt.subplot(2, 3, 1)
    meas_noises = [r['measurement_noise'] for r in results_list]
    rmses = [r['rmse'] for r in results_list]
    ax1.plot(meas_noises, rmses, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Measurement Noise', fontsize=12)
    ax1.set_ylabel('RMSE (m)', fontsize=12)
    ax1.set_title('RMSE vs Measurement Noise', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Mark optimal
    min_idx = np.argmin(rmses)
    ax1.plot(meas_noises[min_idx], rmses[min_idx], 'r*', markersize=15, label='Optimal')
    ax1.legend()
    
    # Plot 2: Convergence Time vs Measurement Noise
    ax2 = plt.subplot(2, 3, 2)
    conv_times = [r['convergence_time'] for r in results_list]
    ax2.plot(meas_noises, conv_times, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Measurement Noise', fontsize=12)
    ax2.set_ylabel('Convergence Time (s)', fontsize=12)
    ax2.set_title('Convergence Speed', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Tracking Accuracy vs Measurement Noise
    ax3 = plt.subplot(2, 3, 3)
    tracking = [r['tracking_accuracy'] for r in results_list]
    ax3.plot(meas_noises, tracking, 'mo-', linewidth=2, markersize=8)
    ax3.set_xlabel('Measurement Noise', fontsize=12)
    ax3.set_ylabel('Tracking Accuracy (mean error, m)', fontsize=12)
    ax3.set_title('Tracking Accuracy', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Stability vs Measurement Noise
    ax4 = plt.subplot(2, 3, 4)
    stabilities = [r['stability'] for r in results_list]
    ax4.plot(meas_noises, stabilities, 'ro-', linewidth=2, markersize=8)
    ax4.set_xlabel('Measurement Noise', fontsize=12)
    ax4.set_ylabel('Stability (std of error, m)', fontsize=12)
    ax4.set_title('Error Stability', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5-6: Error trajectories for all simulations
    colors = plt.cm.plasma(np.linspace(0, 1, len(results_list)))
    
    ax5 = plt.subplot(2, 3, (5, 6))
    for i, r in enumerate(results_list):
        ax5.plot(r['times'], r['errors'], color=colors[i], 
                linewidth=1.5, label=f"MN={r['measurement_noise']:.3f}", alpha=0.7)
    
    ax5.set_xlabel('Time (s)', fontsize=12)
    ax5.set_ylabel('Position Error (m)', fontsize=12)
    ax5.set_title('Error Evolution for Different Measurement Noise Values', 
                 fontsize=14, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'task_b2_measurement_noise_variation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    # Use optimal process noise from B1 (you can adjust this based on B1 results)
    results = task_b2_measurement_noise_variation(
        optimal_process_noise_xy=1e-4,
        optimal_process_noise_theta=1e-4
    )
    print("\nTask B2 complete!")

