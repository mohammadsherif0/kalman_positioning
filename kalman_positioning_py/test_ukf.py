#!/usr/bin/env python3
"""
Test script for UKF implementation

This script tests the basic functionality of the UKF without ROS.
"""

import numpy as np
import sys
import os

# Add parent directory to path for module import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kalman_positioning_py.ukf import UKF
from kalman_positioning_py.landmark_manager import LandmarkManager


def test_landmark_manager():
    """Test landmark manager."""
    print("\n" + "="*60)
    print("Testing Landmark Manager")
    print("="*60)
    
    lm = LandmarkManager()
    
    # Try to load landmarks
    landmarks_file = os.path.join(os.path.dirname(__file__), '..', 'landmarks.csv')
    if not os.path.exists(landmarks_file):
        print(f"Warning: Landmarks file not found at {landmarks_file}")
        return False
    
    success = lm.load_from_csv(landmarks_file)
    if not success:
        print("Failed to load landmarks")
        return False
    
    print(f"✓ Loaded {lm.get_num_landmarks()} landmarks")
    
    # Test queries
    if lm.has_landmark(0):
        pos = lm.get_landmark(0)
        print(f"✓ Landmark 0 at position: {pos}")
    
    # Test spatial query
    nearby = lm.get_landmarks_in_radius(0, 0, 5.0)
    print(f"✓ Found {len(nearby)} landmarks within 5m of origin")
    
    return True


def test_ukf_basic():
    """Test basic UKF functionality."""
    print("\n" + "="*60)
    print("Testing UKF Basic Functionality")
    print("="*60)
    
    # Create UKF instance
    ukf = UKF(
        process_noise_xy=0.1,
        process_noise_theta=0.05,
        measurement_noise_xy=0.2,
        num_landmarks=100,
        decomposition='svd'
    )
    
    print(f"✓ UKF initialized with state: {ukf.get_state()}")
    print(f"✓ Initial covariance diagonal: {np.diag(ukf.get_covariance())}")
    
    # Test sigma point generation
    sigma_points = ukf.generate_sigma_points(ukf.x, ukf.P)
    print(f"✓ Generated {len(sigma_points)} sigma points")
    print(f"  First sigma point: {sigma_points[0]}")
    
    # Test process model
    state_pred = ukf.process_model(ukf.x, dt=0.1, dx=0.5, dy=0.2, dtheta=0.1)
    print(f"✓ Process model output: {state_pred}")
    
    # Test prediction
    ukf.predict(dt=0.1, dx=0.5, dy=0.2, dtheta=0.1)
    print(f"✓ After prediction: position=({ukf.get_position()}), theta={ukf.get_orientation():.3f}")
    
    return True


def test_ukf_with_landmarks():
    """Test UKF with landmark observations."""
    print("\n" + "="*60)
    print("Testing UKF with Landmark Observations")
    print("="*60)
    
    # Create UKF
    ukf = UKF(
        process_noise_xy=0.1,
        process_noise_theta=0.05,
        measurement_noise_xy=0.2,
        num_landmarks=100,
        decomposition='svd'
    )
    
    # Set some test landmarks
    landmarks = {
        0: (5.0, 0.0),
        1: (0.0, 5.0),
        2: (-5.0, 0.0),
    }
    ukf.set_landmarks(landmarks)
    print(f"✓ Set {len(landmarks)} test landmarks")
    
    # Set initial state: robot at origin
    ukf.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Test measurement model
    z_pred = ukf.measurement_model(ukf.x, landmark_id=0)
    print(f"✓ Predicted measurement for landmark 0: {z_pred}")
    print(f"  (Should be approximately [5.0, 0.0] in robot frame)")
    
    # Simulate some observations (with noise)
    observations = [
        (0, 4.9, 0.1),   # Landmark 0
        (1, 0.2, 4.8),   # Landmark 1
    ]
    
    print(f"✓ Simulating {len(observations)} landmark observations")
    ukf.update(observations)
    
    print(f"✓ After update: position=({ukf.get_position()}), theta={ukf.get_orientation():.3f}")
    
    return True


def test_decomposition_comparison():
    """Compare Cholesky vs SVD decomposition."""
    print("\n" + "="*60)
    print("Comparing Decomposition Methods")
    print("="*60)
    
    # Test both methods
    for method in ['cholesky', 'svd']:
        print(f"\nTesting {method.upper()} decomposition:")
        
        ukf = UKF(
            process_noise_xy=0.1,
            process_noise_theta=0.05,
            measurement_noise_xy=0.2,
            num_landmarks=100,
            decomposition=method
        )
        
        # Run a prediction
        ukf.predict(dt=0.1, dx=1.0, dy=0.5, dtheta=0.2)
        
        print(f"  ✓ State: {ukf.get_state()}")
        print(f"  ✓ Covariance det: {np.linalg.det(ukf.get_covariance()):.6f}")
    
    return True


def test_convergence_simulation():
    """Simulate robot motion with landmark observations."""
    print("\n" + "="*60)
    print("Simulating Robot Motion with UKF")
    print("="*60)
    
    # Create UKF
    ukf = UKF(
        process_noise_xy=0.1,
        process_noise_theta=0.05,
        measurement_noise_xy=0.2,
        num_landmarks=100,
        decomposition='svd'
    )
    
    # Set landmarks in a square pattern
    landmarks = {
        0: (10.0, 0.0),
        1: (10.0, 10.0),
        2: (0.0, 10.0),
        3: (-10.0, 10.0),
    }
    ukf.set_landmarks(landmarks)
    
    # True robot trajectory (circular motion)
    true_trajectory = []
    estimated_trajectory = []
    
    print("\nSimulating 20 time steps...")
    for t in range(20):
        # True motion
        true_x = 5.0 * np.cos(t * 0.1)
        true_y = 5.0 * np.sin(t * 0.1)
        true_theta = t * 0.1 + np.pi/2
        
        # Noisy odometry
        dx = 0.5 * np.cos(t * 0.1) + np.random.normal(0, 0.05)
        dy = 0.5 * np.sin(t * 0.1) + np.random.normal(0, 0.05)
        dtheta = 0.1 + np.random.normal(0, 0.02)
        
        # Prediction
        ukf.predict(dt=0.1, dx=dx, dy=dy, dtheta=dtheta)
        
        # Simulate landmark observations every 5 steps
        if t % 5 == 0:
            observations = []
            for lid, (lx, ly) in landmarks.items():
                # Calculate true relative position
                dx_world = lx - true_x
                dy_world = ly - true_y
                cos_theta = np.cos(true_theta)
                sin_theta = np.sin(true_theta)
                x_rel = cos_theta * dx_world + sin_theta * dy_world + np.random.normal(0, 0.2)
                y_rel = -sin_theta * dx_world + cos_theta * dy_world + np.random.normal(0, 0.2)
                observations.append((lid, x_rel, y_rel))
            
            ukf.update(observations)
        
        # Store trajectories
        true_trajectory.append([true_x, true_y])
        est_x, est_y = ukf.get_position()
        estimated_trajectory.append([est_x, est_y])
        
        if t % 5 == 0:
            error = np.sqrt((true_x - est_x)**2 + (true_y - est_y)**2)
            print(f"  Step {t:2d}: True=({true_x:6.2f}, {true_y:6.2f}), "
                  f"Est=({est_x:6.2f}, {est_y:6.2f}), Error={error:.3f}m")
    
    # Calculate final RMSE
    true_trajectory = np.array(true_trajectory)
    estimated_trajectory = np.array(estimated_trajectory)
    errors = np.sqrt(np.sum((true_trajectory - estimated_trajectory)**2, axis=1))
    rmse = np.sqrt(np.mean(errors**2))
    
    print(f"\n✓ Simulation complete")
    print(f"  Final RMSE: {rmse:.3f}m")
    print(f"  Max error: {np.max(errors):.3f}m")
    print(f"  Mean error: {np.mean(errors):.3f}m")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("UKF Implementation Test Suite")
    print("="*60)
    
    tests = [
        ("Landmark Manager", test_landmark_manager),
        ("UKF Basic", test_ukf_basic),
        ("UKF with Landmarks", test_ukf_with_landmarks),
        ("Decomposition Comparison", test_decomposition_comparison),
        ("Convergence Simulation", test_convergence_simulation),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {name} test PASSED")
            else:
                failed += 1
                print(f"\n✗ {name} test FAILED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {name} test FAILED with exception:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

