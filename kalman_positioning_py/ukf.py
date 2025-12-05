"""Lightweight Unscented Kalman Filter for 2D pose with landmark observations."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


class UKF:
    """UKF with state [x, y, theta, vx, vy]."""

    ALPHA = 1e-3
    BETA = 2.0
    KAPPA = 0.0

    def __init__(
        self,
        process_noise_xy: float,
        process_noise_theta: float,
        measurement_noise_xy: float,
        num_landmarks: int,
        decomposition: str = "svd",
    ) -> None:
        self.nx = 5
        self.nz = 2
        self.decomposition = decomposition

        self.lambda_ = self.ALPHA**2 * (self.nx + self.KAPPA) - self.nx
        self.gamma = math.sqrt(self.nx + self.lambda_)

        self.x = np.zeros(self.nx)
        self.P = np.eye(self.nx)

        self.Q = np.diag(
            [
                process_noise_xy,
                process_noise_xy,
                process_noise_theta,
                0.0,
                0.0,
            ]
        )
        self.R = np.diag([measurement_noise_xy, measurement_noise_xy])

        self.Wm, self.Wc = self._compute_weights()
        self.landmarks: Dict[int, Tuple[float, float]] = {}

    def _compute_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        wm = np.zeros(2 * self.nx + 1)
        wc = np.zeros(2 * self.nx + 1)
        wm[0] = self.lambda_ / (self.nx + self.lambda_)
        wc[0] = wm[0] + (1 - self.ALPHA**2 + self.BETA)
        for i in range(1, 2 * self.nx + 1):
            wm[i] = 1.0 / (2 * (self.nx + self.lambda_))
            wc[i] = wm[i]
        return wm, wc

    def set_landmarks(self, landmarks: Dict[int, Tuple[float, float]]) -> None:
        self.landmarks = landmarks

    def has_landmark(self, landmark_id: int) -> bool:
        return landmark_id in self.landmarks

    def generate_sigma_points(
        self, mean: np.ndarray, cov: np.ndarray
    ) -> List[np.ndarray]:
        sigma_points: List[np.ndarray] = [mean.copy()]

        scaled_cov = (self.nx + self.lambda_) * cov
        L = None
        if self.decomposition == "cholesky":
            try:
                L = np.linalg.cholesky(scaled_cov)
            except np.linalg.LinAlgError:
                # Fallback to SVD if Cholesky fails
                self.decomposition = "svd"
        if L is None:
            # SVD-based square root
            U, S, _ = np.linalg.svd(scaled_cov)
            L = U @ np.diag(np.sqrt(S))

        for i in range(self.nx):
            col = L[:, i]
            sigma_points.append(mean + col)
            sigma_points.append(mean - col)

        return sigma_points

    def process_model(
        self, state: np.ndarray, dt: float, dx: float, dy: float, dtheta: float
    ) -> np.ndarray:
        new_state = state.copy()
        new_state[0] += dx
        new_state[1] += dy
        new_state[2] = self._normalize_angle(state[2] + dtheta)
        if dt > 1e-6:
            new_state[3] = dx / dt
            new_state[4] = dy / dt
        return new_state

    def measurement_model(self, state: np.ndarray, landmark_id: int) -> np.ndarray:
        if landmark_id not in self.landmarks:
            return np.zeros(self.nz)
        lx, ly = self.landmarks[landmark_id]
        x, y, theta = state[0], state[1], state[2]
        dx = lx - x
        dy = ly - y
        c, s = math.cos(theta), math.sin(theta)
        rel_x = c * dx + s * dy
        rel_y = -s * dx + c * dy
        return np.array([rel_x, rel_y])

    def predict(self, dt: float, dx: float, dy: float, dtheta: float) -> None:
        sigma_points = self.generate_sigma_points(self.x, self.P)

        propagated = [
            self.process_model(sp, dt=dt, dx=dx, dy=dy, dtheta=dtheta)
            for sp in sigma_points
        ]

        x_pred = np.zeros(self.nx)
        for w, sp in zip(self.Wm, propagated):
            x_pred += w * sp

        P_pred = np.zeros((self.nx, self.nx))
        for w, sp in zip(self.Wc, propagated):
            diff = sp - x_pred
            diff[2] = self._normalize_angle(diff[2])
            P_pred += w * np.outer(diff, diff)

        P_pred += self.Q

        self.x = x_pred
        self.P = P_pred

    def update(self, observations: Iterable[Tuple[int, float, float]]) -> None:
        for landmark_id, z_x, z_y in observations:
            if landmark_id not in self.landmarks:
                continue

            sigma_points = self.generate_sigma_points(self.x, self.P)
            Z_sigma = [
                self.measurement_model(sp, landmark_id) for sp in sigma_points
            ]

            z_pred = np.zeros(self.nz)
            for w, z in zip(self.Wm, Z_sigma):
                z_pred += w * z

            S = np.zeros((self.nz, self.nz))
            T = np.zeros((self.nx, self.nz))
            for w, sp, z in zip(self.Wc, sigma_points, Z_sigma):
                z_diff = z - z_pred
                x_diff = sp - self.x
                x_diff[2] = self._normalize_angle(x_diff[2])
                S += w * np.outer(z_diff, z_diff)
                T += w * np.outer(x_diff, z_diff)

            S += self.R

            try:
                K = T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                # If inversion fails, skip this measurement
                continue

            z_meas = np.array([z_x, z_y])
            innovation = z_meas - z_pred

            self.x = self.x + K @ innovation
            self.x[2] = self._normalize_angle(self.x[2])
            self.P = self.P - K @ S @ K.T
            # Symmetrize to avoid numerical drift
            self.P = 0.5 * (self.P + self.P.T)

    # Convenience accessors
    def get_state(self) -> np.ndarray:
        return self.x

    def get_covariance(self) -> np.ndarray:
        return self.P

    def get_position(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    def get_orientation(self) -> float:
        return float(self.x[2])

    def get_velocity(self) -> Tuple[float, float]:
        return float(self.x[3]), float(self.x[4])

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
"""
Unscented Kalman Filter (UKF) for Robot Localization

State vector: [x, y, theta, vx, vy]
  - x, y: position (m)
  - theta: orientation (rad)
  - vx, vy: velocity (m/s)

Measurement: [x_rel, y_rel] - relative landmark position in robot frame
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional


class UKF:
    """Unscented Kalman Filter for robot localization with landmarks."""
    
    # UKF parameters
    ALPHA = 1e-3  # Spread of sigma points
    BETA = 2.0    # Distribution info (Gaussian optimal)
    KAPPA = 0.0   # Secondary scaling parameter
    
    def __init__(self, process_noise_xy: float, process_noise_theta: float,
                 measurement_noise_xy: float, num_landmarks: int, 
                 decomposition: str = 'svd'):
        """
        Initialize the Unscented Kalman Filter.
        
        Task A1: Constructor and Initialization
        
        Args:
            process_noise_xy: Process noise for position (m)
            process_noise_theta: Process noise for orientation (rad)
            measurement_noise_xy: Measurement noise for landmark observations (m)
            num_landmarks: Number of landmarks (for reference)
            decomposition: Method for covariance decomposition ('cholesky' or 'svd')
        """
        # State dimensions
        self.nx = 5  # State dimension [x, y, theta, vx, vy]
        self.nz = 2  # Measurement dimension [x_rel, y_rel]
        
        # Decomposition method
        self.decomposition = decomposition
        
        # Calculate UKF parameters
        self.lambda_ = self.ALPHA**2 * (self.nx + self.KAPPA) - self.nx
        self.gamma = math.sqrt(self.nx + self.lambda_)
        
        # Number of sigma points
        self.n_sigma = 2 * self.nx + 1
        
        # Calculate weights for mean and covariance
        self.Wm = np.zeros(self.n_sigma)
        self.Wc = np.zeros(self.n_sigma)
        
        # Weight for first sigma point (mean)
        self.Wm[0] = self.lambda_ / (self.nx + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.nx + self.lambda_) + (1 - self.ALPHA**2 + self.BETA)
        
        # Weights for other sigma points
        for i in range(1, self.n_sigma):
            self.Wm[i] = 1.0 / (2.0 * (self.nx + self.lambda_))
            self.Wc[i] = 1.0 / (2.0 * (self.nx + self.lambda_))
        
        # Initialize state: [x, y, theta, vx, vy]
        self.x = np.zeros(self.nx)
        
        # Initialize state covariance (identity matrix)
        self.P = np.eye(self.nx)
        
        # Process noise covariance Q
        # Noise for [x, y, theta, vx, vy]
        # Velocity is derived from position, so we set noise to 0
        self.Q = np.diag([
            process_noise_xy**2,      # x
            process_noise_xy**2,      # y
            process_noise_theta**2,   # theta
            0.0,                      # vx (derived)
            0.0                       # vy (derived)
        ])
        
        # Measurement noise covariance R
        self.R = np.diag([
            measurement_noise_xy**2,  # x_rel
            measurement_noise_xy**2   # y_rel
        ])
        
        # Landmark positions (to be set later)
        self.landmarks: Dict[int, Tuple[float, float]] = {}
        
        print(f"UKF initialized with {decomposition} decomposition")
        print(f"  State dim: {self.nx}, Measurement dim: {self.nz}")
        print(f"  Lambda: {self.lambda_:.4f}, Gamma: {self.gamma:.4f}")
        print(f"  Process noise (xy, theta): {process_noise_xy:.4f}, {process_noise_theta:.4f}")
        print(f"  Measurement noise: {measurement_noise_xy:.4f}")
    
    def generate_sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Generate sigma points from mean and covariance.
        
        Task A2: Sigma Point Generation
        
        Args:
            mean: State mean vector (nx,)
            cov: State covariance matrix (nx, nx)
            
        Returns:
            Sigma points array (n_sigma, nx)
        """
        n = len(mean)
        sigma_points = np.zeros((self.n_sigma, n))
        
        # First sigma point is the mean
        sigma_points[0] = mean
        
        # Compute matrix square root of covariance
        # We'll try both methods: Cholesky and SVD
        try:
            if self.decomposition == 'cholesky':
                # Cholesky decomposition: P = L * L^T
                # Ensure P is positive definite by adding small value to diagonal
                cov_reg = cov + np.eye(n) * 1e-9
                L = np.linalg.cholesky(cov_reg)
                sqrt_matrix = L
            else:  # SVD or eigenvalue decomposition
                # SVD: P = U * S * V^T, sqrt(P) = U * sqrt(S)
                # For symmetric positive definite matrix, use eigendecomposition
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                # Ensure non-negative eigenvalues
                eigenvalues = np.maximum(eigenvalues, 1e-9)
                sqrt_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues))
                
        except np.linalg.LinAlgError:
            print("Warning: Covariance decomposition failed, using eigendecomposition")
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.maximum(eigenvalues, 1e-9)
            sqrt_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        
        # Generate sigma points symmetrically around the mean
        scaled_sqrt = self.gamma * sqrt_matrix
        
        for i in range(n):
            # Positive direction
            sigma_points[i + 1] = mean + scaled_sqrt[:, i]
            # Negative direction
            sigma_points[n + i + 1] = mean - scaled_sqrt[:, i]
        
        return sigma_points
    
    def process_model(self, state: np.ndarray, dt: float, 
                     dx: float, dy: float, dtheta: float) -> np.ndarray:
        """
        Apply motion model to a state vector.
        
        Task A3: Process Model
        
        Args:
            state: Current state [x, y, theta, vx, vy]
            dt: Time step (s)
            dx, dy: Linear displacement in x and y (m)
            dtheta: Angular displacement (rad)
            
        Returns:
            New state after applying motion model
        """
        new_state = state.copy()
        
        # Update position
        new_state[0] = state[0] + dx  # x' = x + dx
        new_state[1] = state[1] + dy  # y' = y + dy
        
        # Update orientation and normalize to [-pi, pi]
        new_state[2] = self.normalize_angle(state[2] + dtheta)
        
        # Update velocities (derived from displacement and time)
        if dt > 1e-6:  # Avoid division by zero
            new_state[3] = dx / dt  # vx = dx / dt
            new_state[4] = dy / dt  # vy = dy / dt
        else:
            new_state[3] = 0.0
            new_state[4] = 0.0
        
        return new_state
    
    def measurement_model(self, state: np.ndarray, landmark_id: int) -> Optional[np.ndarray]:
        """
        Predict measurement given current state and landmark.
        
        Task A4: Measurement Model
        
        Args:
            state: Current state [x, y, theta, vx, vy]
            landmark_id: ID of the landmark
            
        Returns:
            Predicted measurement [x_rel, y_rel] in robot frame, or None if landmark not found
        """
        if landmark_id not in self.landmarks:
            return None
        
        # Get landmark position in world frame
        lx, ly = self.landmarks[landmark_id]
        
        # Get robot state
        x, y, theta = state[0], state[1], state[2]
        
        # Calculate relative position in world frame
        dx_world = lx - x
        dy_world = ly - y
        
        # Transform to robot frame (rotate by -theta)
        # Robot frame: x-axis points forward, y-axis points left
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        x_rel = cos_theta * dx_world + sin_theta * dy_world
        y_rel = -sin_theta * dx_world + cos_theta * dy_world
        
        return np.array([x_rel, y_rel])
    
    def predict(self, dt: float, dx: float, dy: float, dtheta: float):
        """
        Kalman Filter Prediction Step (Time Update).
        
        Task A5: Predict Step
        
        Args:
            dt: Time step (s)
            dx, dy: Linear displacement from odometry (m)
            dtheta: Angular displacement from odometry (rad)
        """
        # Generate sigma points from current state and covariance
        sigma_points = self.generate_sigma_points(self.x, self.P)
        
        # Propagate each sigma point through the process model
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(self.n_sigma):
            sigma_points_pred[i] = self.process_model(
                sigma_points[i], dt, dx, dy, dtheta
            )
        
        # Calculate predicted mean
        x_pred = np.zeros(self.nx)
        for i in range(self.n_sigma):
            x_pred += self.Wm[i] * sigma_points_pred[i]
        
        # Normalize angle in predicted mean
        x_pred[2] = self.normalize_angle(x_pred[2])
        
        # Calculate predicted covariance
        P_pred = np.zeros((self.nx, self.nx))
        for i in range(self.n_sigma):
            diff = sigma_points_pred[i] - x_pred
            # Normalize angle difference
            diff[2] = self.normalize_angle(diff[2])
            P_pred += self.Wc[i] * np.outer(diff, diff)
        
        # Add process noise
        P_pred += self.Q
        
        # Update state and covariance
        self.x = x_pred
        self.P = P_pred
    
    def update(self, landmark_observations: List[Tuple[int, float, float]]):
        """
        Kalman Filter Update Step (Measurement Update).
        
        Task A6: Update Step
        
        Args:
            landmark_observations: List of (landmark_id, x_rel, y_rel) tuples
                                  where x_rel, y_rel are observed relative positions
        """
        if not landmark_observations:
            return
        
        # Process each landmark observation
        for landmark_id, obs_x, obs_y in landmark_observations:
            if landmark_id not in self.landmarks:
                continue
            
            # Generate sigma points
            sigma_points = self.generate_sigma_points(self.x, self.P)
            
            # Transform sigma points through measurement model
            z_sigma = []
            for i in range(self.n_sigma):
                z_pred = self.measurement_model(sigma_points[i], landmark_id)
                if z_pred is not None:
                    z_sigma.append(z_pred)
                else:
                    z_sigma.append(np.zeros(self.nz))
            z_sigma = np.array(z_sigma)  # (n_sigma, nz)
            
            # Calculate predicted measurement mean
            z_mean = np.zeros(self.nz)
            for i in range(self.n_sigma):
                z_mean += self.Wm[i] * z_sigma[i]
            
            # Calculate innovation covariance (measurement covariance)
            P_zz = np.zeros((self.nz, self.nz))
            for i in range(self.n_sigma):
                diff_z = z_sigma[i] - z_mean
                P_zz += self.Wc[i] * np.outer(diff_z, diff_z)
            
            # Add measurement noise
            P_zz += self.R
            
            # Calculate cross-covariance
            P_xz = np.zeros((self.nx, self.nz))
            for i in range(self.n_sigma):
                diff_x = sigma_points[i] - self.x
                diff_x[2] = self.normalize_angle(diff_x[2])  # Normalize angle
                diff_z = z_sigma[i] - z_mean
                P_xz += self.Wc[i] * np.outer(diff_x, diff_z)
            
            # Calculate Kalman gain
            try:
                K = P_xz @ np.linalg.inv(P_zz)
            except np.linalg.LinAlgError:
                print("Warning: Singular matrix in Kalman gain calculation")
                continue
            
            # Innovation (measurement residual)
            z_obs = np.array([obs_x, obs_y])
            innovation = z_obs - z_mean
            
            # Update state
            self.x = self.x + K @ innovation
            self.x[2] = self.normalize_angle(self.x[2])  # Normalize angle
            
            # Update covariance
            self.P = self.P - K @ P_zz @ K.T
            
            # Ensure P remains symmetric and positive definite
            self.P = 0.5 * (self.P + self.P.T)
            self.P += np.eye(self.nx) * 1e-9
    
    def set_landmarks(self, landmarks: Dict[int, Tuple[float, float]]):
        """Set landmark positions."""
        self.landmarks = landmarks.copy()
    
    def has_landmark(self, landmark_id: int) -> bool:
        """Check if landmark exists."""
        return landmark_id in self.landmarks
    
    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        return self.x.copy()
    
    def get_covariance(self) -> np.ndarray:
        """Get current state covariance."""
        return self.P.copy()
    
    def get_position(self) -> Tuple[float, float]:
        """Get position estimate (x, y)."""
        return (float(self.x[0]), float(self.x[1]))
    
    def get_orientation(self) -> float:
        """Get orientation estimate (theta)."""
        return float(self.x[2])
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get velocity estimate (vx, vy)."""
        return (float(self.x[3]), float(self.x[4]))
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

