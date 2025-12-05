"""
STUDENT ASSIGNMENT: Unscented Kalman Filter Implementation

This file contains the complete UKF implementation for robot localization.
Students should implement each method according to the UKF algorithm.

Reference: Wan, E. A., & Van Der Merwe, R. (2000). 
"The Unscented Kalman Filter for Nonlinear Estimation"
"""

import numpy as np
from typing import Dict, List, Tuple


class UKF:
    """
    Unscented Kalman Filter for robot localization with landmarks
    State: [x, y, theta, vx, vy]
    """
    
    # UKF parameters (constants)
    # Note: Alpha=0.1, Beta=2.0, Kappa=0 are standard values
    # These result in lambda=-4.95, gamma=0.224 for n=5 dimensions
    ALPHA = 0.1   # Spread of sigma points (0.001 to 1, larger = more spread)
    BETA = 2.0    # Distribution info (Gaussian = 2.0)
    KAPPA = 0.0   # Secondary scaling parameter
    
    def __init__(self, process_noise_xy: float, process_noise_theta: float,
                 measurement_noise_xy: float, num_landmarks: int = 0):
        """
        Initialize the Unscented Kalman Filter
        
        Task A1: UKF Constructor and Initialization (10 Points)
        
        STUDENT TODO:
        1. Initialize filter parameters (alpha, beta, kappa, lambda)
        2. Initialize state vector x_ with zeros
        3. Initialize state covariance matrix P_ 
        4. Set process noise covariance Q_
        5. Set measurement noise covariance R_
        6. Calculate sigma point weights for mean and covariance
        
        Args:
            process_noise_xy: Process noise for x, y positions
            process_noise_theta: Process noise for orientation
            measurement_noise_xy: Measurement noise for landmark observations
            num_landmarks: Number of landmarks (optional, for reference)
        """
        # STUDENT IMPLEMENTATION - Task A1
        # ====================================================================
        
        # State dimension (x, y, theta, vx, vy)
        self.nx = 5
        
        # Measurement dimension (x, y relative to robot)
        self.nz = 2
        
        # Initialize state vector: [x, y, theta, vx, vy]
        self.x = np.zeros(self.nx)
        
        # Initialize state covariance matrix with high initial uncertainty
        # Since we start at [0,0,0,0,0] but don't know true position
        self.P = np.diag([100.0, 100.0, 10.0, 1.0, 1.0])  # High position uncertainty initially
        
        # Process noise covariance Q
        # [x, y, theta, vx, vy]
        self.Q = np.diag([
            process_noise_xy,      # x position noise
            process_noise_xy,      # y position noise
            process_noise_theta,   # theta orientation noise
            0.0,                   # vx velocity (determined by motion)
            0.0                    # vy velocity (determined by motion)
        ])
        
        # Measurement noise covariance R (2x2 for x, y observations)
        self.R = np.diag([
            measurement_noise_xy,  # x observation noise
            measurement_noise_xy   # y observation noise
        ])
        
        # Calculate UKF parameters
        # lambda = alpha^2 * (n + kappa) - n
        self.lambda_ = self.ALPHA**2 * (self.nx + self.KAPPA) - self.nx
        
        # gamma = sqrt(n + lambda)
        self.gamma = np.sqrt(self.nx + self.lambda_)
        
        # Calculate weights for mean (Wm) and covariance (Wc)
        self.Wm = np.zeros(2 * self.nx + 1)
        self.Wc = np.zeros(2 * self.nx + 1)
        
        # Weight for mean of first sigma point
        self.Wm[0] = self.lambda_ / (self.nx + self.lambda_)
        
        # Weight for covariance of first sigma point
        self.Wc[0] = self.lambda_ / (self.nx + self.lambda_) + (1 - self.ALPHA**2 + self.BETA)
        
        # Weights for remaining sigma points (same for mean and covariance)
        for i in range(1, 2 * self.nx + 1):
            self.Wm[i] = 1.0 / (2.0 * (self.nx + self.lambda_))
            self.Wc[i] = 1.0 / (2.0 * (self.nx + self.lambda_))
        
        # Landmark storage
        self.landmarks: Dict[int, Tuple[float, float]] = {}
        
        print(f"UKF Initialized:")
        print(f"  State dimension: {self.nx}")
        print(f"  Lambda: {self.lambda_:.6f}")
        print(f"  Gamma: {self.gamma:.6f}")
        print(f"  Weights Wm: {self.Wm}")
        print(f"  Weights Wc: {self.Wc}")
    
    def generate_sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> List[np.ndarray]:
        """
        Generate sigma points from mean and covariance
        
        Task A2: Sigma Point Generation (8 Points)
        
        STUDENT TODO:
        1. Start with the mean as the first sigma point
        2. Compute Cholesky decomposition of covariance
        3. Generate 2*n symmetric sigma points around the mean
        
        Args:
            mean: State mean vector (nx,)
            cov: State covariance matrix (nx, nx)
            
        Returns:
            List of 2*n+1 sigma points, each of dimension (nx,)
        """
        # STUDENT IMPLEMENTATION - Task A2
        # ====================================================================
        
        n = len(mean)
        sigma_points = []
        
        # First sigma point is the mean itself
        sigma_points.append(mean.copy())
        
        # Compute matrix square root using eigenvalue decomposition
        # This is more robust than Cholesky and doesn't require positive definiteness
        # P = U * D * U^T, so sqrt(P) = U * sqrt(D) * U^T
        
        # Scale covariance matrix
        scaled_cov = (n + self.lambda_) * cov
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(scaled_cov)
        
        # Ensure positive eigenvalues (add small epsilon if needed)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        
        # Compute matrix square root: sqrt(P) = U * sqrt(D)
        sqrt_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        
        # Generate 2*n sigma points using columns of sqrt_matrix
        for i in range(n):
            # Positive direction: mean + column i of sqrt_matrix
            sigma_points.append(mean + sqrt_matrix[:, i])
            
        for i in range(n):
            # Negative direction: mean - column i of sqrt_matrix
            sigma_points.append(mean - sqrt_matrix[:, i])
        
        return sigma_points
    
    def process_model(self, state: np.ndarray, dt: float,
                      dx: float, dy: float, dtheta: float) -> np.ndarray:
        """
        Apply motion model to a state vector
        
        Task A3: Process Model (7 Points)
        
        STUDENT TODO:
        1. Updates position: x' = x + dx, y' = y + dy
        2. Updates orientation: theta' = theta + dtheta (normalized)
        3. Updates velocities: vx' = dx/dt, vy' = dy/dt
        
        Args:
            state: Current state [x, y, theta, vx, vy]
            dt: Time step (seconds)
            dx, dy: Linear displacement in x, y
            dtheta: Angular displacement
            
        Returns:
            New state after applying motion model
        """
        # STUDENT IMPLEMENTATION - Task A3
        # ====================================================================
        
        new_state = state.copy()
        
        # Update position
        new_state[0] = state[0] + dx  # x' = x + dx
        new_state[1] = state[1] + dy  # y' = y + dy
        
        # Update orientation and normalize to [-pi, pi]
        new_state[2] = self.normalize_angle(state[2] + dtheta)
        
        # Update velocities (dx/dt, dy/dt)
        if dt > 1e-6:  # Avoid division by zero
            new_state[3] = dx / dt  # vx
            new_state[4] = dy / dt  # vy
        else:
            new_state[3] = 0.0
            new_state[4] = 0.0
        
        return new_state
    
    def measurement_model(self, state: np.ndarray, landmark_id: int) -> np.ndarray:
        """
        Predict measurement given current state and landmark
        
        Task A4: Measurement Model (8 Points)
        
        STUDENT TODO:
        1. Calculate relative position: landmark - robot position
        2. Transform to robot frame using robot orientation
        3. Return relative position in robot frame
        
        Args:
            state: Current state [x, y, theta, vx, vy]
            landmark_id: ID of the landmark to observe
            
        Returns:
            Predicted measurement [rel_x, rel_y] in robot frame
        """
        # STUDENT IMPLEMENTATION - Task A4
        # ====================================================================
        
        # Check if landmark exists
        if landmark_id not in self.landmarks:
            return np.zeros(2)
        
        # Get landmark position in world frame
        lx, ly = self.landmarks[landmark_id]
        
        # Get robot position and orientation
        rx, ry, theta = state[0], state[1], state[2]
        
        # Calculate relative position in world frame
        dx_world = lx - rx
        dy_world = ly - ry
        
        # Transform to robot frame using rotation
        # [rel_x]   [cos(theta)  sin(theta)] [dx_world]
        # [rel_y] = [-sin(theta) cos(theta)] [dy_world]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        rel_x = cos_theta * dx_world + sin_theta * dy_world
        rel_y = -sin_theta * dx_world + cos_theta * dy_world
        
        return np.array([rel_x, rel_y])
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
    
    def predict(self, dt: float, dx: float, dy: float, dtheta: float):
        """
        Kalman Filter Prediction Step (Time Update)
        
        Task A5: Predict Step (10 Points)
        
        STUDENT TODO:
        1. Generate sigma points from current state and covariance
        2. Propagate each sigma point through motion model
        3. Calculate mean and covariance of predicted sigma points
        4. Add process noise
        5. Update state and covariance estimates
        
        Args:
            dt: Time step (seconds)
            dx, dy: Linear displacement
            dtheta: Angular displacement
        """
        # STUDENT IMPLEMENTATION - Task A5
        # ====================================================================
        
        # Step 1: Generate sigma points from current state and covariance
        sigma_points = self.generate_sigma_points(self.x, self.P)
        
        # Step 2: Propagate each sigma point through the process model
        sigma_points_pred = []
        for sp in sigma_points:
            sp_pred = self.process_model(sp, dt, dx, dy, dtheta)
            sigma_points_pred.append(sp_pred)
        
        # Step 3: Calculate predicted mean
        x_pred = np.zeros(self.nx)
        for i, sp in enumerate(sigma_points_pred):
            x_pred += self.Wm[i] * sp
        
        # Normalize angle in predicted mean
        x_pred[2] = self.normalize_angle(x_pred[2])
        
        # Step 4: Calculate predicted covariance
        P_pred = np.zeros((self.nx, self.nx))
        for i, sp in enumerate(sigma_points_pred):
            # Calculate deviation
            diff = sp - x_pred
            # Normalize angle difference
            diff[2] = self.normalize_angle(diff[2])
            # Add weighted outer product
            P_pred += self.Wc[i] * np.outer(diff, diff)
        
        # Step 5: Add process noise
        P_pred += self.Q
        
        # Update state and covariance
        self.x = x_pred
        self.P = P_pred
    
    def update(self, landmark_observations: List[Tuple[int, float, float, float]]):
        """
        Kalman Filter Update Step (Measurement Update)
        
        Task A6: Update Step (7 Points)
        
        STUDENT TODO:
        1. Generate sigma points
        2. Transform through measurement model
        3. Calculate predicted measurement mean
        4. Calculate measurement and cross-covariance
        5. Compute Kalman gain
        6. Update state with innovation
        7. Update covariance
        
        Args:
            landmark_observations: List of (id, x, y, noise_cov) tuples
                - id: landmark ID
                - x, y: observed position in robot frame
                - noise_cov: observation noise covariance (not used, we use R)
        """
        # STUDENT IMPLEMENTATION - Task A6
        # ====================================================================
        
        if not landmark_observations:
            return
        
        # Process each landmark observation
        for obs in landmark_observations:
            landmark_id, obs_x, obs_y, _ = obs
            
            # Skip if landmark is not known
            if landmark_id not in self.landmarks:
                continue
            
            # Step 1: Generate sigma points from current state
            sigma_points = self.generate_sigma_points(self.x, self.P)
            
            # Step 2: Transform sigma points through measurement model
            Z_sigma = []
            for sp in sigma_points:
                z_pred = self.measurement_model(sp, landmark_id)
                Z_sigma.append(z_pred)
            
            # Step 3: Calculate predicted measurement mean
            z_mean = np.zeros(self.nz)
            for i, z in enumerate(Z_sigma):
                z_mean += self.Wm[i] * z
            
            # Step 4a: Calculate measurement covariance (innovation covariance)
            P_zz = np.zeros((self.nz, self.nz))
            for i, z in enumerate(Z_sigma):
                diff = z - z_mean
                P_zz += self.Wc[i] * np.outer(diff, diff)
            
            # Add measurement noise
            P_zz += self.R
            
            # Step 4b: Calculate cross-covariance
            P_xz = np.zeros((self.nx, self.nz))
            for i, (sp, z) in enumerate(zip(sigma_points, Z_sigma)):
                x_diff = sp - self.x
                x_diff[2] = self.normalize_angle(x_diff[2])  # Normalize angle
                z_diff = z - z_mean
                P_xz += self.Wc[i] * np.outer(x_diff, z_diff)
            
            # Step 5: Compute Kalman gain
            try:
                K = P_xz @ np.linalg.inv(P_zz)
            except np.linalg.LinAlgError:
                print("Warning: Singular matrix in Kalman gain computation")
                continue
            
            # Step 6: Update state with innovation
            z_obs = np.array([obs_x, obs_y])
            innovation = z_obs - z_mean
            self.x = self.x + K @ innovation
            
            # Normalize angle after update
            self.x[2] = self.normalize_angle(self.x[2])
            
            # Step 7: Update covariance
            self.P = self.P - K @ P_zz @ K.T
            
            # Ensure P remains symmetric and positive definite
            self.P = (self.P + self.P.T) / 2.0
    
    def set_landmarks(self, landmarks: Dict[int, Tuple[float, float]]):
        """Set landmark positions"""
        self.landmarks = landmarks
    
    def has_landmark(self, landmark_id: int) -> bool:
        """Check if landmark exists"""
        return landmark_id in self.landmarks
    
    def get_state(self) -> np.ndarray:
        """Get estimated state [x, y, theta, vx, vy]"""
        return self.x.copy()
    
    def get_covariance(self) -> np.ndarray:
        """Get state covariance matrix"""
        return self.P.copy()
    
    def get_position(self) -> Tuple[float, float]:
        """Get position estimate (x, y)"""
        return (self.x[0], self.x[1])
    
    def get_orientation(self) -> float:
        """Get orientation estimate (theta)"""
        return self.x[2]
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get velocity estimate (vx, vy)"""
        return (self.x[3], self.x[4])

