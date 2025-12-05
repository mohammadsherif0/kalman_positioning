#!/usr/bin/env python3
"""
Task B1: Process Noise Variation (10 Points)

Run 5 simulations varying process noise with constant measurement noise:
- Measure RMSE, convergence speed, and stability
- Plot RMSE vs. process noise
- Document optimal parameters and explain behavior
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from run_simulation import run_simulation


def task_b1_process_noise_variation():
    """
    Experiment with varying process noise while keeping measurement noise constant
    """
    print("=" * 80)
    print("TASK B1: Process Noise Variation Experiment")
    print("=" * 80)
    
    # Fixed parameters
    duration = 40.0
    dt = 0.1
    measurement_noise_xy = 0.01  # Fixed measurement noise
    odometry_noise_xy = 0.1
    odometry_noise_theta = 0.05
    observation_noise = 0.1
    observation_radius = 5.0
    
    # Varying process noise (5 different values)
    process_noise_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    process_noise_theta_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    
    results_list = []
    
    print(f"\nFixed Parameters:")
    print(f"  Measurement Noise: {measurement_noise_xy}")
    print(f"  Duration: {duration}s")
    print(f"  Observation Radius: {observation_radius}m")
    print(f"\nRunning {len(process_noise_values)} simulations...\n")
    
    # Run simulations with different process noise values
    for i, (pn_xy, pn_theta) in enumerate(zip(process_noise_values, process_noise_theta_values)):
        print(f"\n{'='*70}")
        print(f"Simulation {i+1}/{len(process_noise_values)}")
        print(f"Process Noise (XY): {pn_xy}, Process Noise (Theta): {pn_theta}")
        print(f"{'='*70}")
        
        results = run_simulation(
            duration=duration,
            dt=dt,
            process_noise_xy=pn_xy,
            process_noise_theta=pn_theta,
            measurement_noise_xy=measurement_noise_xy,
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
        
        results_list.append({
            'process_noise_xy': pn_xy,
            'process_noise_theta': pn_theta,
            'rmse': results['rmse'],
            'mean_error': results['mean_error'],
            'max_error': results['max_error'],
            'convergence_time': convergence_time,
            'stability': stability,
            'errors': errors,
            'times': results['times']
        })
        
        print(f"\nResults:")
        print(f"  RMSE: {results['rmse']:.4f} m")
        print(f"  Convergence Time: {convergence_time:.2f} s")
        print(f"  Stability (std): {stability:.4f} m")
    
    # Generate comprehensive report
    print("\n" + "=" * 80)
    print("TASK B1: Summary Report")
    print("=" * 80)
    print("\nProcess Noise Variation Results:")
    print("-" * 80)
    print(f"{'Process Noise':<20} {'RMSE (m)':<12} {'Conv. Time (s)':<15} {'Stability (m)':<15}")
    print("-" * 80)
    
    for r in results_list:
        print(f"{r['process_noise_xy']:<20.2e} {r['rmse']:<12.4f} "
              f"{r['convergence_time']:<15.2f} {r['stability']:<15.4f}")
    
    # Find optimal parameters
    min_rmse_idx = np.argmin([r['rmse'] for r in results_list])
    optimal = results_list[min_rmse_idx]
    
    print("\n" + "-" * 80)
    print("Optimal Parameters:")
    print(f"  Process Noise (XY): {optimal['process_noise_xy']:.2e}")
    print(f"  Process Noise (Theta): {optimal['process_noise_theta']:.2e}")
    print(f"  RMSE: {optimal['rmse']:.4f} m")
    print(f"  Convergence Time: {optimal['convergence_time']:.2f} s")
    print(f"  Stability: {optimal['stability']:.4f} m")
    
    print("\nBehavior Analysis:")
    print("-" * 80)
    print("1. Low Process Noise (1e-6 to 1e-5):")
    print("   - Filter trusts prediction more, less responsive to measurements")
    print("   - May be slower to converge if initial state is far from truth")
    print("   - More stable but potentially biased if model is imperfect")
    
    print("\n2. Moderate Process Noise (1e-4 to 1e-3):")
    print("   - Balanced trust between prediction and measurement")
    print("   - Faster convergence and good tracking performance")
    print("   - Optimal for most scenarios")
    
    print("\n3. High Process Noise (1e-2):")
    print("   - Filter trusts measurements more than prediction")
    print("   - Fast convergence but more sensitive to measurement noise")
    print("   - May be less stable and noisier estimates")
    
    # Create visualizations
    create_b1_plots(results_list)
    
    return results_list


def create_b1_plots(results_list):
    """Create visualization plots for Task B1"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: RMSE vs Process Noise
    ax1 = plt.subplot(2, 3, 1)
    process_noises = [r['process_noise_xy'] for r in results_list]
    rmses = [r['rmse'] for r in results_list]
    ax1.semilogx(process_noises, rmses, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Process Noise (XY)', fontsize=12)
    ax1.set_ylabel('RMSE (m)', fontsize=12)
    ax1.set_title('RMSE vs Process Noise', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Mark optimal
    min_idx = np.argmin(rmses)
    ax1.plot(process_noises[min_idx], rmses[min_idx], 'r*', markersize=15, label='Optimal')
    ax1.legend()
    
    # Plot 2: Convergence Time vs Process Noise
    ax2 = plt.subplot(2, 3, 2)
    conv_times = [r['convergence_time'] for r in results_list]
    ax2.semilogx(process_noises, conv_times, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Process Noise (XY)', fontsize=12)
    ax2.set_ylabel('Convergence Time (s)', fontsize=12)
    ax2.set_title('Convergence Speed', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Stability vs Process Noise
    ax3 = plt.subplot(2, 3, 3)
    stabilities = [r['stability'] for r in results_list]
    ax3.semilogx(process_noises, stabilities, 'ro-', linewidth=2, markersize=8)
    ax3.set_xlabel('Process Noise (XY)', fontsize=12)
    ax3.set_ylabel('Stability (std of error, m)', fontsize=12)
    ax3.set_title('Error Stability', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4-6: Error trajectories for all simulations
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_list)))
    
    ax4 = plt.subplot(2, 1, 2)
    for i, r in enumerate(results_list):
        ax4.plot(r['times'], r['errors'], color=colors[i], 
                linewidth=1.5, label=f"PN={r['process_noise_xy']:.1e}", alpha=0.7)
    
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylabel('Position Error (m)', fontsize=12)
    ax4.set_title('Error Evolution for Different Process Noise Values', 
                 fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'task_b1_process_noise_variation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    results = task_b1_process_noise_variation()
    print("\nTask B1 complete!")

