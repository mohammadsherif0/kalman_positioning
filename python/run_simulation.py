#!/usr/bin/env python3
"""
Main simulation script for UKF robot localization

This script demonstrates the UKF implementation by simulating a robot
moving in a circular trajectory with noisy odometry and landmark observations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kalman_positioning import UKF, LandmarkManager
from kalman_positioning.simulator import RobotSimulator, CircularTrajectory, generate_grid_landmarks


def run_simulation(duration: float = 30.0,
                   dt: float = 0.1,
                   process_noise_xy: float = 1e-4,
                   process_noise_theta: float = 1e-4,
                   measurement_noise_xy: float = 0.01,
                   odometry_noise_xy: float = 0.1,
                   odometry_noise_theta: float = 0.05,
                   observation_noise: float = 0.1,
                   observation_radius: float = 5.0,
                   visualize: bool = True,
                   save_plot: str = None):
    """
    Run UKF simulation
    
    Args:
        duration: Simulation duration (seconds)
        dt: Time step (seconds)
        process_noise_xy: UKF process noise for x, y
        process_noise_theta: UKF process noise for theta
        measurement_noise_xy: UKF measurement noise
        odometry_noise_xy: Simulator odometry noise
        odometry_noise_theta: Simulator orientation noise
        observation_noise: Simulator landmark observation noise
        observation_radius: Maximum landmark observation distance
        visualize: Whether to show visualization
        save_plot: Path to save plot (optional)
        
    Returns:
        Dictionary with simulation results
    """
    print("=" * 70)
    print("UKF Robot Localization Simulation")
    print("=" * 70)
    
    # Generate landmarks in a grid
    landmarks = generate_grid_landmarks(grid_size=10, spacing=2.0)
    print(f"\nGenerated {len(landmarks)} landmarks in grid pattern")
    
    # Initialize simulator
    simulator = RobotSimulator(
        landmarks=landmarks,
        odometry_noise_xy=odometry_noise_xy,
        odometry_noise_theta=odometry_noise_theta,
        observation_noise=observation_noise,
        observation_radius=observation_radius
    )
    
    # Initialize UKF
    ukf = UKF(
        process_noise_xy=process_noise_xy,
        process_noise_theta=process_noise_theta,
        measurement_noise_xy=measurement_noise_xy,
        num_landmarks=len(landmarks)
    )
    ukf.set_landmarks(landmarks)
    
    # Initialize trajectory (circular motion)
    trajectory = CircularTrajectory(
        center_x=10.0,
        center_y=10.0,
        radius=5.0,
        angular_velocity=0.3
    )
    
    # Reset robot to starting position
    simulator.reset(x=15.0, y=10.0, theta=np.pi)
    
    # Storage for results
    times = []
    true_positions = []
    estimated_positions = []
    noisy_positions = []
    errors = []
    
    noisy_x, noisy_y = 15.0, 10.0
    
    print(f"\nSimulation Parameters:")
    print(f"  Duration: {duration}s")
    print(f"  Time step: {dt}s")
    print(f"  Process noise (xy): {process_noise_xy}")
    print(f"  Process noise (theta): {process_noise_theta}")
    print(f"  Measurement noise: {measurement_noise_xy}")
    print(f"  Odometry noise (xy): {odometry_noise_xy}")
    print(f"  Observation noise: {observation_noise}")
    print(f"  Observation radius: {observation_radius}m")
    
    print("\nRunning simulation...")
    
    # Simulation loop
    num_steps = int(duration / dt)
    for step in range(num_steps):
        t = step * dt
        
        # Get motion command
        dx, dy, dtheta = trajectory.get_motion(dt)
        
        # Move robot (true motion)
        simulator.move(dx, dy, dtheta, dt)
        
        # Get noisy odometry
        noisy_dx, noisy_dy, noisy_dtheta, _ = simulator.get_noisy_odometry(dx, dy, dtheta, dt)
        
        # Update noisy dead-reckoning position
        noisy_x += noisy_dx
        noisy_y += noisy_dy
        
        # UKF Prediction step
        ukf.predict(dt, noisy_dx, noisy_dy, noisy_dtheta)
        
        # Get landmark observations
        observations = simulator.get_landmark_observations()
        
        # UKF Update step
        if observations:
            ukf.update(observations)
        
        # Store results
        true_x, true_y, _, _, _ = simulator.get_true_state()
        est_x, est_y = ukf.get_position()
        
        times.append(t)
        true_positions.append((true_x, true_y))
        estimated_positions.append((est_x, est_y))
        noisy_positions.append((noisy_x, noisy_y))
        
        # Calculate error
        error = np.sqrt((true_x - est_x)**2 + (true_y - est_y)**2)
        errors.append(error)
        
        # Print progress
        if step % 100 == 0:
            print(f"  Step {step}/{num_steps}, t={t:.1f}s, "
                  f"True: ({true_x:.2f}, {true_y:.2f}), "
                  f"Est: ({est_x:.2f}, {est_y:.2f}), "
                  f"Error: {error:.3f}m, "
                  f"Obs: {len(observations)}")
    
    # Calculate statistics
    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors**2))
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    print("\n" + "=" * 70)
    print("Simulation Results:")
    print("=" * 70)
    print(f"  RMSE: {rmse:.4f} m")
    print(f"  Mean Error: {mean_error:.4f} m")
    print(f"  Max Error: {max_error:.4f} m")
    print(f"  Final Error: {errors[-1]:.4f} m")
    
    # Visualization
    if visualize or save_plot:
        visualize_results(times, true_positions, estimated_positions, 
                         noisy_positions, errors, landmarks, save_plot)
    
    return {
        'times': times,
        'true_positions': true_positions,
        'estimated_positions': estimated_positions,
        'noisy_positions': noisy_positions,
        'errors': errors,
        'rmse': rmse,
        'mean_error': mean_error,
        'max_error': max_error,
        'landmarks': landmarks
    }


def visualize_results(times, true_positions, estimated_positions, 
                     noisy_positions, errors, landmarks, save_path=None):
    """Visualize simulation results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Extract coordinates
    true_x = [p[0] for p in true_positions]
    true_y = [p[1] for p in true_positions]
    est_x = [p[0] for p in estimated_positions]
    est_y = [p[1] for p in estimated_positions]
    noisy_x = [p[0] for p in noisy_positions]
    noisy_y = [p[1] for p in noisy_positions]
    
    # Plot 1: Trajectories
    ax1 = axes[0, 0]
    
    # Plot landmarks
    lm_x = [lm[0] for lm in landmarks.values()]
    lm_y = [lm[1] for lm in landmarks.values()]
    ax1.scatter(lm_x, lm_y, c='gray', marker='s', s=100, 
               alpha=0.5, label='Landmarks', zorder=1)
    
    # Plot trajectories
    ax1.plot(true_x, true_y, 'g-', linewidth=2, label='True', alpha=0.8)
    ax1.plot(noisy_x, noisy_y, 'r--', linewidth=1, label='Noisy Odometry', alpha=0.6)
    ax1.plot(est_x, est_y, 'b-', linewidth=2, label='UKF Estimate', alpha=0.8)
    
    # Mark start and end
    ax1.plot(true_x[0], true_y[0], 'go', markersize=10, label='Start')
    ax1.plot(true_x[-1], true_y[-1], 'ro', markersize=10, label='End')
    
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title('Robot Trajectories', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Position Error over Time
    ax2 = axes[0, 1]
    ax2.plot(times, errors, 'b-', linewidth=1.5)
    ax2.axhline(y=np.mean(errors), color='r', linestyle='--', 
               label=f'Mean: {np.mean(errors):.3f}m')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Position Error (m)', fontsize=12)
    ax2.set_title('UKF Position Error Over Time', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: X Position Comparison
    ax3 = axes[1, 0]
    ax3.plot(times, true_x, 'g-', linewidth=2, label='True', alpha=0.8)
    ax3.plot(times, noisy_x, 'r--', linewidth=1, label='Noisy', alpha=0.6)
    ax3.plot(times, est_x, 'b-', linewidth=2, label='Estimated', alpha=0.8)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('X Position (m)', fontsize=12)
    ax3.set_title('X Position Over Time', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Y Position Comparison
    ax4 = axes[1, 1]
    ax4.plot(times, true_y, 'g-', linewidth=2, label='True', alpha=0.8)
    ax4.plot(times, noisy_y, 'r--', linewidth=1, label='Noisy', alpha=0.6)
    ax4.plot(times, est_y, 'b-', linewidth=2, label='Estimated', alpha=0.8)
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylabel('Y Position (m)', fontsize=12)
    ax4.set_title('Y Position Over Time', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    if not save_path:  # Only show if not saving
        plt.show()


if __name__ == "__main__":
    # Run simulation with default parameters
    results = run_simulation(
        duration=30.0,
        dt=0.1,
        visualize=True
    )
    
    print("\nSimulation complete!")

