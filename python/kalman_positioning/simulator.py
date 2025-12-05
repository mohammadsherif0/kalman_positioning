"""
Simulation Framework for UKF Robot Localization

This module provides a simple simulation environment to test the UKF
without needing ROS2. It simulates:
- Robot motion (circular trajectory)
- Noisy odometry measurements
- Noisy landmark observations
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import time


class RobotSimulator:
    """Simulates a robot moving in a 2D environment with landmarks"""
    
    def __init__(self, landmarks: Dict[int, Tuple[float, float]],
                 odometry_noise_xy: float = 0.1,
                 odometry_noise_theta: float = 0.05,
                 observation_noise: float = 0.1,
                 observation_radius: float = 5.0):
        """
        Initialize robot simulator
        
        Args:
            landmarks: Dictionary of landmark_id -> (x, y)
            odometry_noise_xy: Standard deviation of odometry noise (m)
            odometry_noise_theta: Standard deviation of orientation noise (rad)
            observation_noise: Standard deviation of landmark observation noise (m)
            observation_radius: Maximum distance to observe landmarks (m)
        """
        self.landmarks = landmarks
        self.odometry_noise_xy = odometry_noise_xy
        self.odometry_noise_theta = odometry_noise_theta
        self.observation_noise = observation_noise
        self.observation_radius = observation_radius
        
        # True robot state
        self.true_x = 0.0
        self.true_y = 0.0
        self.true_theta = 0.0
        self.true_vx = 0.0
        self.true_vy = 0.0
        
        # Previous state for odometry calculation
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_theta = 0.0
        self.prev_time = time.time()
    
    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        """Reset robot to initial state"""
        self.true_x = x
        self.true_y = y
        self.true_theta = theta
        self.true_vx = 0.0
        self.true_vy = 0.0
        self.prev_x = x
        self.prev_y = y
        self.prev_theta = theta
        self.prev_time = time.time()
    
    def move(self, dx: float, dy: float, dtheta: float, dt: float):
        """
        Move the robot (true motion without noise)
        
        Args:
            dx, dy: Linear displacement in world frame
            dtheta: Angular displacement
            dt: Time step
        """
        self.true_x += dx
        self.true_y += dy
        self.true_theta += dtheta
        self.true_theta = self._normalize_angle(self.true_theta)
        
        if dt > 0:
            self.true_vx = dx / dt
            self.true_vy = dy / dt
    
    def get_noisy_odometry(self, dx: float, dy: float, dtheta: float, dt: float) -> Tuple[float, float, float, float]:
        """
        Get noisy odometry measurement
        
        Args:
            dx, dy: True linear displacement
            dtheta: True angular displacement
            dt: Time step
            
        Returns:
            (noisy_dx, noisy_dy, noisy_dtheta, dt)
        """
        # Add Gaussian noise to odometry
        noisy_dx = dx + np.random.normal(0, self.odometry_noise_xy)
        noisy_dy = dy + np.random.normal(0, self.odometry_noise_xy)
        noisy_dtheta = dtheta + np.random.normal(0, self.odometry_noise_theta)
        
        return (noisy_dx, noisy_dy, noisy_dtheta, dt)
    
    def get_landmark_observations(self) -> List[Tuple[int, float, float, float]]:
        """
        Get noisy landmark observations in robot frame
        
        Returns:
            List of (landmark_id, rel_x, rel_y, noise_cov) tuples
        """
        observations = []
        
        for landmark_id, (lx, ly) in self.landmarks.items():
            # Calculate distance to landmark
            distance = np.sqrt((lx - self.true_x)**2 + (ly - self.true_y)**2)
            
            # Only observe landmarks within observation radius
            if distance > self.observation_radius:
                continue
            
            # Calculate true relative position in world frame
            dx_world = lx - self.true_x
            dy_world = ly - self.true_y
            
            # Transform to robot frame
            cos_theta = np.cos(self.true_theta)
            sin_theta = np.sin(self.true_theta)
            
            rel_x = cos_theta * dx_world + sin_theta * dy_world
            rel_y = -sin_theta * dx_world + cos_theta * dy_world
            
            # Add noise
            noisy_rel_x = rel_x + np.random.normal(0, self.observation_noise)
            noisy_rel_y = rel_y + np.random.normal(0, self.observation_noise)
            
            observations.append((landmark_id, noisy_rel_x, noisy_rel_y, self.observation_noise**2))
        
        return observations
    
    def get_true_state(self) -> Tuple[float, float, float, float, float]:
        """Get true robot state"""
        return (self.true_x, self.true_y, self.true_theta, self.true_vx, self.true_vy)
    
    def get_true_position(self) -> Tuple[float, float]:
        """Get true robot position"""
        return (self.true_x, self.true_y)
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle


class CircularTrajectory:
    """Generates circular trajectory for robot simulation"""
    
    def __init__(self, center_x: float = 5.0, center_y: float = 5.0,
                 radius: float = 3.0, angular_velocity: float = 0.2):
        """
        Initialize circular trajectory
        
        Args:
            center_x, center_y: Center of circle
            radius: Radius of circle
            angular_velocity: Angular velocity (rad/s)
        """
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.angular_velocity = angular_velocity
        self.time = 0.0
    
    def get_motion(self, dt: float) -> Tuple[float, float, float]:
        """
        Get motion command for time step
        
        Args:
            dt: Time step
            
        Returns:
            (dx, dy, dtheta) in world frame
        """
        # Current angle
        angle = self.angular_velocity * self.time
        
        # Next angle
        next_angle = self.angular_velocity * (self.time + dt)
        
        # Current position
        x = self.center_x + self.radius * np.cos(angle)
        y = self.center_y + self.radius * np.sin(angle)
        
        # Next position
        next_x = self.center_x + self.radius * np.cos(next_angle)
        next_y = self.center_y + self.radius * np.sin(next_angle)
        
        # Displacement
        dx = next_x - x
        dy = next_y - y
        
        # Angular displacement (tangent to circle)
        dtheta = self.angular_velocity * dt
        
        # Update time
        self.time += dt
        
        return (dx, dy, dtheta)
    
    def reset(self):
        """Reset trajectory"""
        self.time = 0.0


def generate_grid_landmarks(grid_size: int = 10, spacing: float = 2.0) -> Dict[int, Tuple[float, float]]:
    """
    Generate landmarks in a grid pattern
    
    Args:
        grid_size: Number of landmarks per side
        spacing: Distance between landmarks
        
    Returns:
        Dictionary of landmark_id -> (x, y)
    """
    landmarks = {}
    landmark_id = 0
    
    for i in range(grid_size):
        for j in range(grid_size):
            x = i * spacing
            y = j * spacing
            landmarks[landmark_id] = (x, y)
            landmark_id += 1
    
    return landmarks

