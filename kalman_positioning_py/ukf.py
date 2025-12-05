"""
UKF Implementation for Robot Localization
Tasks A1-A6
"""
import numpy as np


class UKF:
    """Unscented Kalman Filter for robot localization with landmarks."""
    
    def __init__(self, process_noise_xy, process_noise_theta, measurement_noise_xy):
        """
        Task A1: UKF Constructor and Initialization
        
        Initialize:
        - state x = [0, 0, 0, 0, 0]^T
        - Q = diag(process_noise_xy, process_noise_xy, process_noise_theta, 0, 0)
        - R = diag(measurement_noise_xy, measurement_noise_xy)
        - P as identity matrix
        - UKF parameters: λ, γ, and weights
        """
        # State dimension (x, y, theta, vx, vy)
        self.n = 5
        
        # Initialize state x = [0, 0, 0, 0, 0]^T
        self.x = np.zeros(5)
        
        # Initialize P as identity matrix
        self.P = np.eye(5)
        
        # Set Q = diag(process_noise_xy, process_noise_xy, process_noise_theta, 0, 0)
        self.Q = np.diag([process_noise_xy, process_noise_xy, process_noise_theta, 0.0, 0.0])
        
        # Set R = diag(measurement_noise_xy, measurement_noise_xy)
        self.R = np.diag([measurement_noise_xy, measurement_noise_xy])
        
        # Calculate UKF parameters: λ, γ, and weights
        alpha = 0.001
        beta = 2.0
        kappa = 0.0
        
        self.lambda_ = alpha**2 * (self.n + kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lambda_)
        
        # Weights for mean
        self.Wm = np.zeros(2 * self.n + 1)
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        for i in range(1, 2 * self.n + 1):
            self.Wm[i] = 1.0 / (2.0 * (self.n + self.lambda_))
        
        # Weights for covariance
        self.Wc = np.zeros(2 * self.n + 1)
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1.0 - alpha**2 + beta)
        for i in range(1, 2 * self.n + 1):
            self.Wc[i] = 1.0 / (2.0 * (self.n + self.lambda_))
        
        # Landmarks dictionary
        self.landmarks = {}
    
    def set_landmarks(self, landmarks_dict):
        """Set landmark positions {id: (x, y)}."""
        self.landmarks = landmarks_dict
    
    def generate_sigma_points(self, mean, cov):
        """
        Task A2: Sigma Point Generation
        
        - Calculate Cholesky decomposition of covariance
        - Generate 2n+1 sigma points: X0 = x, Xi = x ± γLi
        - Return vector of sigma points
        """
        sigma_points = []
        
        # X0 = x
        sigma_points.append(mean.copy())
        
        # Cholesky decomposition: P = L * L^T
        try:
            L = np.linalg.cholesky((self.n + self.lambda_) * cov)
        except np.linalg.LinAlgError:
            # Fallback to eigenvalue decomposition if Cholesky fails
            eigvals, eigvecs = np.linalg.eigh((self.n + self.lambda_) * cov)
            eigvals = np.maximum(eigvals, 1e-10)
            L = eigvecs @ np.diag(np.sqrt(eigvals))
        
        # Generate Xi = x + γLi and Xi = x - γLi
        for i in range(self.n):
            sigma_points.append(mean + L[:, i])
            sigma_points.append(mean - L[:, i])
        
        return sigma_points
    
    def process_model(self, state, dt, dx, dy, dtheta):
        """
        Task A3: Process Model
        
        - Update position: x' = x + Δx, y' = y + Δy
        - Update orientation: θ' = θ + Δθ (normalize to [-π, π])
        - Calculate velocities: vx = Δx / Δt, vy = Δy / Δt
        """
        new_state = np.zeros(5)
        
        # Update position
        new_state[0] = state[0] + dx
        new_state[1] = state[1] + dy
        
        # Update orientation (normalize to [-π, π])
        new_state[2] = self.normalize_angle(state[2] + dtheta)
        
        # Calculate velocities
        if dt > 1e-6:
            new_state[3] = dx / dt
            new_state[4] = dy / dt
        else:
            new_state[3] = 0.0
            new_state[4] = 0.0
        
        return new_state
    
    def measurement_model(self, state, landmark_id):
        """
        Task A4: Measurement Model
        
        - Find landmark position (lx, ly) using landmark_id
        - Calculate relative position in robot frame
        """
        if landmark_id not in self.landmarks:
            return np.zeros(2)
        
        # Get landmark position
        lx, ly = self.landmarks[landmark_id]
        
        # Robot state
        rx, ry, theta = state[0], state[1], state[2]
        
        # Calculate relative position in world frame
        dx_world = lx - rx
        dy_world = ly - ry
        
        # Transform to robot frame
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        rel_x = cos_theta * dx_world + sin_theta * dy_world
        rel_y = -sin_theta * dx_world + cos_theta * dy_world
        
        return np.array([rel_x, rel_y])
    
    def predict(self, dt, dx, dy, dtheta):
        """
        Task A5: Predict Step
        
        - Generate sigma points from (x, P)
        - Propagate through process model
        - Calculate weighted mean and covariance
        - Add process noise: P += Q
        """
        # Generate sigma points
        sigma_points = self.generate_sigma_points(self.x, self.P)
        
        # Propagate through process model
        sigma_points_pred = []
        for sp in sigma_points:
            sigma_points_pred.append(self.process_model(sp, dt, dx, dy, dtheta))
        
        # Calculate weighted mean
        x_pred = np.zeros(self.n)
        for i, sp in enumerate(sigma_points_pred):
            x_pred += self.Wm[i] * sp
        x_pred[2] = self.normalize_angle(x_pred[2])
        
        # Calculate weighted covariance
        P_pred = np.zeros((self.n, self.n))
        for i, sp in enumerate(sigma_points_pred):
            diff = sp - x_pred
            diff[2] = self.normalize_angle(diff[2])
            P_pred += self.Wc[i] * np.outer(diff, diff)
        
        # Add process noise
        P_pred += self.Q
        
        # Update state and covariance
        self.x = x_pred
        self.P = P_pred
    
    def update(self, landmark_observations):
        """
        Task A6: Update Step
        
        For each landmark observation:
        - Generate sigma points and transform through measurement model
        - Calculate ẑ, P_zz, and P_xz
        - Compute Kalman gain: K = P_xz P_zz^-1
        - Update state and covariance
        """
        print(f"\n=== UKF UPDATE (state: {self.x[0]:.2f}, {self.x[1]:.2f}, {self.x[2]:.2f}) ===")
        
        for obs in landmark_observations:
            landmark_id, obs_x, obs_y = obs[0], obs[1], obs[2]
            
            if landmark_id not in self.landmarks:
                continue
            
            # Debug: show landmark world position and what we observe
            lm_world = self.landmarks[landmark_id]
            predicted_obs = self.measurement_model(self.x, landmark_id)
            print(f"  LM{landmark_id} world=({lm_world[0]:.1f},{lm_world[1]:.1f}) pred=({predicted_obs[0]:.2f},{predicted_obs[1]:.2f}) obs=({obs_x:.2f},{obs_y:.2f})")
            
            # Generate sigma points
            sigma_points = self.generate_sigma_points(self.x, self.P)
            
            # Transform through measurement model
            Z_sigma = []
            for sp in sigma_points:
                Z_sigma.append(self.measurement_model(sp, landmark_id))
            
            # Calculate predicted measurement mean ẑ
            z_pred = np.zeros(2)
            for i, z in enumerate(Z_sigma):
                z_pred += self.Wm[i] * z
            
            # Calculate P_zz and P_xz
            P_zz = np.zeros((2, 2))
            P_xz = np.zeros((self.n, 2))
            
            for i in range(len(sigma_points)):
                z_diff = Z_sigma[i] - z_pred
                P_zz += self.Wc[i] * np.outer(z_diff, z_diff)
                
                x_diff = sigma_points[i] - self.x
                x_diff[2] = self.normalize_angle(x_diff[2])
                P_xz += self.Wc[i] * np.outer(x_diff, z_diff)
            
            # Add measurement noise
            P_zz += self.R
            
            # Compute Kalman gain
            K = P_xz @ np.linalg.inv(P_zz)
            
            # Update state
            z_obs = np.array([obs_x, obs_y])
            innovation = z_obs - z_pred
            correction = K @ innovation
            print(f"    Innovation: ({innovation[0]:.2f},{innovation[1]:.2f}) -> Correction: dx={correction[0]:.2f}, dy={correction[1]:.2f}")
            
            self.x = self.x + correction
            self.x[2] = self.normalize_angle(self.x[2])
            
            print(f"    New state: ({self.x[0]:.2f}, {self.x[1]:.2f})")
            
            # Update covariance
            self.P = self.P - K @ P_zz @ K.T
    
    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-π, π]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def get_state(self):
        """Return current state estimate."""
        return self.x.copy()
    
    def set_state(self, index, value):
        """Set a state component."""
        self.x[index] = value

