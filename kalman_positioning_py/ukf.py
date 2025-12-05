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
        
        # Initialize P as identity matrix (with high initial uncertainty)
        self.P = np.eye(5) * 100.0  # High uncertainty initially
        
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
        - Calculate relative position: [lx - x, ly - y]
        """
        if landmark_id not in self.landmarks:
            return np.zeros(2)
        
        # Get landmark position
        lx, ly = self.landmarks[landmark_id]
        
        # Robot state
        rx, ry = state[0], state[1]
        
        # Calculate relative position: [lx - x, ly - y]
        # Task says NO rotation - just world frame difference!
        return np.array([lx - rx, ly - ry])
    
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
        Task A6: Update Step - Process ALL landmarks in ONE batch
        
        - Generate sigma points and transform through measurement model
        - Calculate ẑ, P_zz, and P_xz for ALL landmarks together
        - Compute Kalman gain: K = P_xz P_zz^-1
        - Update state and covariance ONCE
        """
        if not landmark_observations:
            return
        
        # Filter valid landmarks
        valid_obs = [(lid, ox, oy) for lid, ox, oy in landmark_observations if lid in self.landmarks]
        if not valid_obs:
            return
        
        n_obs = len(valid_obs)
        nz_total = n_obs * 2
        
        print(f"\n=== UKF UPDATE with {n_obs} landmarks (state: {self.x[0]:.2f}, {self.x[1]:.2f}) ===")
        
        # Generate sigma points ONCE
        sigma_points = self.generate_sigma_points(self.x, self.P)
        
        # Transform ALL sigma points through ALL measurement models
        Z_sigma = []
        for sp in sigma_points:
            z_all = []
            for landmark_id, _, _ in valid_obs:
                z = self.measurement_model(sp, landmark_id)
                z_all.extend([z[0], z[1]])
            Z_sigma.append(np.array(z_all))
        
        # Calculate predicted measurement mean ẑ
        z_pred = np.zeros(nz_total)
        for i, z in enumerate(Z_sigma):
            z_pred += self.Wm[i] * z
        
        # Calculate P_zz and P_xz
        P_zz = np.zeros((nz_total, nz_total))
        P_xz = np.zeros((self.n, nz_total))
        
        for i in range(len(sigma_points)):
            z_diff = Z_sigma[i] - z_pred
            P_zz += self.Wc[i] * np.outer(z_diff, z_diff)
            
            x_diff = sigma_points[i] - self.x
            x_diff[2] = self.normalize_angle(x_diff[2])
            P_xz += self.Wc[i] * np.outer(x_diff, z_diff)
        
        # Add measurement noise (block diagonal)
        for i in range(n_obs):
            P_zz[i*2:i*2+2, i*2:i*2+2] += self.R
        
        # Compute Kalman gain
        K = P_xz @ np.linalg.inv(P_zz)
        
        # Build observation vector
        z_obs = np.zeros(nz_total)
        for i, (_, ox, oy) in enumerate(valid_obs):
            z_obs[i*2] = ox
            z_obs[i*2+1] = oy
        
        # Update state ONCE with ALL observations
        innovation = z_obs - z_pred
        correction = K @ innovation
        
        print(f"  Total innovation norm: {np.linalg.norm(innovation):.2f}")
        print(f"  Position correction: dx={correction[0]:.2f}, dy={correction[1]:.2f}")
        
        self.x = self.x + correction
        self.x[2] = self.normalize_angle(self.x[2])
        
        print(f"  New state: ({self.x[0]:.2f}, {self.x[1]:.2f}, {self.x[2]:.2f})")
        
        # Update covariance ONCE
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

