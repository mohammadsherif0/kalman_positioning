#!/usr/bin/env python3
"""
Master script to run all experimental tasks (B1, B2, B3)

This script runs all three experimental tasks in sequence:
- Task B1: Process Noise Variation
- Task B2: Measurement Noise Variation
- Task B3: Simulation Parameter Variation
"""

import sys
import os
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.task_b1_process_noise import task_b1_process_noise_variation
from experiments.task_b2_measurement_noise import task_b2_measurement_noise_variation
from experiments.task_b3_simulation_params import task_b3_simulation_parameter_variation
import numpy as np


def main():
    """Run all experimental tasks"""
    
    print("=" * 80)
    print("UKF ROBOT LOCALIZATION - COMPLETE EXPERIMENTAL SUITE")
    print("=" * 80)
    print("\nThis script will run all three experimental tasks:")
    print("  - Task B1: Process Noise Variation")
    print("  - Task B2: Measurement Noise Variation")
    print("  - Task B3: Simulation Parameter Variation")
    print("\nTotal estimated time: 5-10 minutes")
    print("=" * 80)
    
    input("\nPress Enter to begin experiments...")
    
    start_time = time.time()
    
    # Task B1: Process Noise Variation
    print("\n\n")
    print("#" * 80)
    print("# TASK B1: PROCESS NOISE VARIATION")
    print("#" * 80)
    
    b1_start = time.time()
    b1_results = task_b1_process_noise_variation()
    b1_duration = time.time() - b1_start
    
    # Extract optimal process noise from B1
    min_rmse_idx = np.argmin([r['rmse'] for r in b1_results])
    optimal_process_noise_xy = b1_results[min_rmse_idx]['process_noise_xy']
    optimal_process_noise_theta = b1_results[min_rmse_idx]['process_noise_theta']
    
    print(f"\nTask B1 completed in {b1_duration:.1f} seconds")
    print(f"Optimal Process Noise: {optimal_process_noise_xy:.2e}")
    
    input("\n\nPress Enter to continue to Task B2...")
    
    # Task B2: Measurement Noise Variation
    print("\n\n")
    print("#" * 80)
    print("# TASK B2: MEASUREMENT NOISE VARIATION")
    print("#" * 80)
    
    b2_start = time.time()
    b2_results = task_b2_measurement_noise_variation(
        optimal_process_noise_xy=optimal_process_noise_xy,
        optimal_process_noise_theta=optimal_process_noise_theta
    )
    b2_duration = time.time() - b2_start
    
    # Extract optimal measurement noise from B2
    min_rmse_idx = np.argmin([r['rmse'] for r in b2_results])
    optimal_measurement_noise = b2_results[min_rmse_idx]['measurement_noise']
    
    print(f"\nTask B2 completed in {b2_duration:.1f} seconds")
    print(f"Optimal Measurement Noise: {optimal_measurement_noise:.3f}")
    
    input("\n\nPress Enter to continue to Task B3...")
    
    # Task B3: Simulation Parameter Variation
    print("\n\n")
    print("#" * 80)
    print("# TASK B3: SIMULATION PARAMETER VARIATION")
    print("#" * 80)
    
    b3_start = time.time()
    b3_results = task_b3_simulation_parameter_variation(
        optimal_process_noise_xy=optimal_process_noise_xy,
        optimal_process_noise_theta=optimal_process_noise_theta,
        optimal_measurement_noise=optimal_measurement_noise
    )
    b3_duration = time.time() - b3_start
    
    print(f"\nTask B3 completed in {b3_duration:.1f} seconds")
    
    # Final summary
    total_duration = time.time() - start_time
    
    print("\n\n")
    print("=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)
    print(f"\nTotal Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"\nTask Durations:")
    print(f"  B1 (Process Noise):     {b1_duration:.1f}s")
    print(f"  B2 (Measurement Noise): {b2_duration:.1f}s")
    print(f"  B3 (Simulation Params): {b3_duration:.1f}s")
    
    print(f"\nOptimal Parameters Found:")
    print(f"  Process Noise (XY):   {optimal_process_noise_xy:.2e}")
    print(f"  Process Noise (Theta): {optimal_process_noise_theta:.2e}")
    print(f"  Measurement Noise:     {optimal_measurement_noise:.3f}")
    
    print(f"\nResults saved to: experiments/results/")
    print("  - task_b1_process_noise_variation.png")
    print("  - task_b2_measurement_noise_variation.png")
    print("  - task_b3_simulation_parameter_variation.png")
    print("  - task_b3_results_table.csv")
    
    print("\n" + "=" * 80)
    print("Thank you for running the UKF experiments!")
    print("=" * 80)


if __name__ == "__main__":
    main()

