#include "kalman_positioning/ukf.hpp"
#include <iostream>
#include <map>

/**
 * STUDENT ASSIGNMENT: Unscented Kalman Filter Implementation
 * 
 * This file contains placeholder implementations for the UKF class methods.
 * Students should implement each method according to the UKF algorithm.
 * 
 * Reference: Wan, E. A., & Van Der Merwe, R. (2000). 
 * "The Unscented Kalman Filter for Nonlinear Estimation"
 */

// ============================================================================
// CONSTRUCTOR
// ============================================================================

/**
 * @brief Initialize the Unscented Kalman Filter
 * 
 * STUDENT TODO:
 * 1. Initialize filter parameters (alpha, beta, kappa, lambda)
 * 2. Initialize state vector x_ with zeros
 * 3. Initialize state covariance matrix P_ 
 * 4. Set process noise covariance Q_
 * 5. Set measurement noise covariance R_
 * 6. Calculate sigma point weights for mean and covariance
 */
UKF::UKF(double process_noise_xy, double process_noise_theta,
         double measurement_noise_xy, int num_landmarks)
    : nx_(5), nz_(2) {
    
    // Task A1: UKF Constructor and Initialization
    // ========================================================================
    
    // 1. Initialize state vector x = [0, 0, 0, 0, 0]
    x_ = Eigen::VectorXd::Zero(nx_);
    
    // 2. Initialize P as identity matrix (Task A1)
    P_ = Eigen::MatrixXd::Identity(nx_, nx_);
    
    // 3. Set Q = diag(process_noise_xy, process_noise_xy, process_noise_theta, 0, 0) (Task A1)
    Q_ = Eigen::MatrixXd::Zero(nx_, nx_);
    Q_(0, 0) = process_noise_xy;      // x
    Q_(1, 1) = process_noise_xy;      // y
    Q_(2, 2) = process_noise_theta;   // theta
    Q_(3, 3) = 0.0;                   // vx (as specified)
    Q_(4, 4) = 0.0;                   // vy (as specified)
    
    // 4. Set measurement noise covariance R
    R_ = Eigen::MatrixXd::Identity(nz_, nz_) * measurement_noise_xy;
    
    // 5. Calculate UKF parameters: λ, γ, and weights (Task A1)
    // Using parameters that ensure positive weights for stability
    double alpha = 1.0;   // Spread parameter
    double beta = 0.0;    // Using 0 instead of 2 to avoid huge W0_c
    double kappa = 2.0;   // Secondary scaling (3-n would be -2, use 2 instead)
    
    lambda_ = alpha * alpha * (nx_ + kappa) - nx_;  // = 1*(5+2) - 5 = 2
    gamma_ = std::sqrt(nx_ + lambda_);               // = sqrt(7) = 2.646
    
    // 6. Calculate weights W^m_i, W^c_i (Task A1)
    Wm_.resize(2 * nx_ + 1);
    Wc_.resize(2 * nx_ + 1);
    
    Wm_[0] = lambda_ / (nx_ + lambda_);                                    // = 2/7 = 0.286
    Wc_[0] = lambda_ / (nx_ + lambda_) + (1.0 - alpha * alpha + beta);   // = 2/7 + 0 = 0.286
    
    for (int i = 1; i < 2 * nx_ + 1; i++) {
        Wm_[i] = 1.0 / (2.0 * (nx_ + lambda_));  // = 1/14 = 0.071
        Wc_[i] = 1.0 / (2.0 * (nx_ + lambda_));  // = 1/14 = 0.071
    }
    
    std::cout << "UKF Initialized: lambda=" << lambda_ << ", gamma=" << gamma_ << std::endl;
}

// ============================================================================
// SIGMA POINT GENERATION
// ============================================================================

/**
 * @brief Generate sigma points from mean and covariance
 * 
 * STUDENT TODO:
 * 1. Start with the mean as the first sigma point
 * 2. Compute Cholesky decomposition of covariance
 * 3. Generate 2*n symmetric sigma points around the mean
 */
std::vector<Eigen::VectorXd> UKF::generateSigmaPoints(const Eigen::VectorXd& mean,
                                                       const Eigen::MatrixXd& cov) {
    // Task A2: Sigma Point Generation (NO CHOLESKY - using eigenvalue decomposition)
    // ========================================================================
    
    std::vector<Eigen::VectorXd> sigma_points;
    int n = mean.size();
    
    // 1. First sigma point is the mean
    sigma_points.push_back(mean);
    
    // 2. Compute matrix square root using eigenvalue decomposition (Task A2)
    // Instead of Cholesky: P = V * D * V^T, so sqrt(P) = V * sqrt(D)
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    Eigen::MatrixXd eigenvectors = solver.eigenvectors();
    
    // Ensure positive eigenvalues for numerical stability
    for (int i = 0; i < eigenvalues.size(); i++) {
        if (eigenvalues(i) < 1e-10) eigenvalues(i) = 1e-10;
    }
    
    // Compute sqrt(P) = V * sqrt(D), then scale by gamma
    // This gives us γ*sqrt(P) as required for sigma points
    Eigen::MatrixXd sqrt_matrix = eigenvectors * eigenvalues.cwiseSqrt().asDiagonal();
    
    // 3. Generate 2*n sigma points: Xi = x ± γ*Li (Task A2)
    for (int i = 0; i < n; i++) {
        sigma_points.push_back(mean + gamma_ * sqrt_matrix.col(i));  // Positive direction
    }
    for (int i = 0; i < n; i++) {
        sigma_points.push_back(mean - gamma_ * sqrt_matrix.col(i));  // Negative direction
    }
    
    return sigma_points;
}

// ============================================================================
// PROCESS MODEL
// ============================================================================

/**
 * @brief Apply motion model to a state vector
 * 
 * STUDENT TODO:
 * 1. Updates position: x' = x + dx, y' = y + dy
 * 2. Updates orientation: theta' = theta + dtheta (normalized)
 * 3. Updates velocities: vx' = dx/dt, vy' = dy/dt
 */
Eigen::VectorXd UKF::processModel(const Eigen::VectorXd& state, double dt,
                                  double dx, double dy, double dtheta) {
    // Task A3: Process Model
    // ========================================================================
    
    Eigen::VectorXd new_state(5);
    
    // Update position: x' = x + dx, y' = y + dy
    new_state(0) = state(0) + dx;
    new_state(1) = state(1) + dy;
    
    // Update orientation: theta' = theta + dtheta (normalized)
    new_state(2) = normalizeAngle(state(2) + dtheta);
    
    // Update velocities: vx = dx/dt, vy = dy/dt
    if (dt > 1e-6) {
        new_state(3) = dx / dt;
        new_state(4) = dy / dt;
    } else {
        new_state(3) = 0.0;
        new_state(4) = 0.0;
    }
    
    return new_state;
}

// ============================================================================
// MEASUREMENT MODEL
// ============================================================================

/**
 * @brief Predict measurement given current state and landmark
 * 
 * STUDENT TODO:
 * 1. Calculate relative position: landmark - robot position
 * 2. Transform to robot frame using robot orientation
 * 3. Return relative position in robot frame
 */
Eigen::Vector2d UKF::measurementModel(const Eigen::VectorXd& state, int landmark_id) {
    // Task A4: Measurement Model
    // ========================================================================
    // The observations from fake_robot are ALREADY in robot frame (relative x, y)
    // So we need to predict what the robot WOULD observe given its state
    
    if (landmarks_.find(landmark_id) == landmarks_.end()) {
        return Eigen::Vector2d::Zero();
    }
    
    // Get landmark position in world frame
    auto landmark = landmarks_[landmark_id];
    double lx = landmark.first;
    double ly = landmark.second;
    
    // Get robot state in world frame
    double rx = state(0);
    double ry = state(1);
    double theta = state(2);
    
    // Calculate relative position in world frame
    double dx_world = lx - rx;
    double dy_world = ly - ry;
    
    // Transform to robot frame (what the robot's sensor would see)
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    
    Eigen::Vector2d predicted_observation;
    // Rotation from world to robot frame
    predicted_observation(0) = cos_theta * dx_world + sin_theta * dy_world;   // forward
    predicted_observation(1) = -sin_theta * dx_world + cos_theta * dy_world;  // sideways
    
    return predicted_observation;
}

// ============================================================================
// ANGLE NORMALIZATION
// ============================================================================

double UKF::normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

// ============================================================================
// PREDICTION STEP
// ============================================================================

/**
 * @brief Kalman Filter Prediction Step (Time Update)
 * 
 * STUDENT TODO:
 * 1. Generate sigma points from current state and covariance
 * 2. Propagate each sigma point through motion model
 * 3. Calculate mean and covariance of predicted sigma points
 * 4. Add process noise
 * 5. Update state and covariance estimates
 */
void UKF::predict(double dt, double dx, double dy, double dtheta) {
    // Task A5: Predict Step
    // ========================================================================
    
    // 1. Generate sigma points from current state and covariance
    std::vector<Eigen::VectorXd> sigma_points = generateSigmaPoints(x_, P_);
    
    // 2. Propagate each sigma point through process model
    std::vector<Eigen::VectorXd> sigma_points_pred;
    for (const auto& sp : sigma_points) {
        sigma_points_pred.push_back(processModel(sp, dt, dx, dy, dtheta));
    }
    
    // 3. Calculate predicted mean
    Eigen::VectorXd x_pred = Eigen::VectorXd::Zero(nx_);
    for (size_t i = 0; i < sigma_points_pred.size(); i++) {
        x_pred += Wm_[i] * sigma_points_pred[i];
    }
    x_pred(2) = normalizeAngle(x_pred(2));  // Normalize angle
    
    // 4. Calculate predicted covariance
    Eigen::MatrixXd P_pred = Eigen::MatrixXd::Zero(nx_, nx_);
    for (size_t i = 0; i < sigma_points_pred.size(); i++) {
        Eigen::VectorXd diff = sigma_points_pred[i] - x_pred;
        diff(2) = normalizeAngle(diff(2));  // Normalize angle difference
        P_pred += Wc_[i] * (diff * diff.transpose());
    }
    
    // 5. Add process noise
    P_pred += Q_;
    
    // Update state and covariance
    x_ = x_pred;
    P_ = P_pred;
}

// ============================================================================
// UPDATE STEP
// ============================================================================

/**
 * @brief Kalman Filter Update Step (Measurement Update)
 * 
 * STUDENT TODO:
 * 1. Generate sigma points
 * 2. Transform through measurement model
 * 3. Calculate predicted measurement mean
 * 4. Calculate measurement and cross-covariance
 * 5. Compute Kalman gain
 * 6. Update state with innovation
 * 7. Update covariance
 */
void UKF::update(const std::vector<std::tuple<int, double, double, double>>& landmark_observations) {
    if (landmark_observations.empty()) {
        return;
    }
    
    // Task A6: Standard UKF Update - One landmark at a time
    // ========================================================================
    
    std::cout << "\n=== UKF UPDATE ===" << std::endl;
    std::cout << "Current state: x=" << x_(0) << ", y=" << x_(1) << ", theta=" << x_(2) << std::endl;
    
    for (const auto& obs : landmark_observations) {
        int landmark_id = std::get<0>(obs);
        double obs_x = std::get<1>(obs);
        double obs_y = std::get<2>(obs);
        
        if (!hasLandmark(landmark_id)) {
            continue;
        }
        
        auto lm = landmarks_[landmark_id];
        std::cout << "  LM" << landmark_id << " world=(" << lm.first << "," << lm.second << ")" << std::endl;
        
        // Debug: Show what we SHOULD observe vs what we DID observe
        Eigen::Vector2d predicted = measurementModel(x_, landmark_id);
        std::cout << "    Predicted obs (robot frame): (" << predicted(0) << "," << predicted(1) << ")" << std::endl;
        std::cout << "    Actual obs    (robot frame): (" << obs_x << "," << obs_y << ")" << std::endl;
        std::cout << "    Innovation: (" << (obs_x - predicted(0)) << "," << (obs_y - predicted(1)) << ")" << std::endl;
        
        // 1. Generate sigma points
        std::vector<Eigen::VectorXd> sigma_points = generateSigmaPoints(x_, P_);
        
        // 2. Transform through measurement model
        std::vector<Eigen::Vector2d> Z_sigma;
        for (const auto& sp : sigma_points) {
            Z_sigma.push_back(measurementModel(sp, landmark_id));
        }
        
        // 3. Calculate predicted measurement mean
        Eigen::Vector2d z_mean = Eigen::Vector2d::Zero();
        for (size_t i = 0; i < Z_sigma.size(); i++) {
            z_mean += Wm_[i] * Z_sigma[i];
        }
        
        // 4. Calculate innovation covariance and cross-covariance
        Eigen::MatrixXd P_zz = Eigen::MatrixXd::Zero(nz_, nz_);
        Eigen::MatrixXd P_xz = Eigen::MatrixXd::Zero(nx_, nz_);
        
        for (size_t i = 0; i < sigma_points.size(); i++) {
            Eigen::Vector2d z_diff = Z_sigma[i] - z_mean;
            P_zz += Wc_[i] * (z_diff * z_diff.transpose());
            
            Eigen::VectorXd x_diff = sigma_points[i] - x_;
            x_diff(2) = normalizeAngle(x_diff(2));
            P_xz += Wc_[i] * (x_diff * z_diff.transpose());
        }
        
        // Add measurement noise
        P_zz += R_;
        
        // 5. Check for numerical issues
        if (P_zz.determinant() < 1e-10) {
            continue;  // Skip this update if covariance is singular
        }
        
        // 6. Compute Kalman gain
        Eigen::MatrixXd K = P_xz * P_zz.inverse();
        
        // 7. Update state with innovation
        Eigen::Vector2d z_obs(obs_x, obs_y);
        Eigen::Vector2d innovation = z_obs - z_mean;
        
        // Limit innovation to prevent huge jumps (optional safety check)
        double innovation_norm = innovation.norm();
        if (innovation_norm > 5.0) {  // Max 5m correction per landmark
            innovation = innovation / innovation_norm * 5.0;
        }
        
        Eigen::VectorXd correction = K * innovation;
        std::cout << "    Kalman gain correction: dx=" << correction(0) << ", dy=" << correction(1) << ", dtheta=" << correction(2) << std::endl;
        
        x_ = x_ + correction;
        x_(2) = normalizeAngle(x_(2));
        
        std::cout << "    New state: x=" << x_(0) << ", y=" << x_(1) << ", theta=" << x_(2) << std::endl;
        
        // 8. Update covariance
        P_ = P_ - K * P_zz * K.transpose();
        P_ = (P_ + P_.transpose()) / 2.0;  // Symmetrize
        
        // Add regularization
        P_ += Eigen::MatrixXd::Identity(nx_, nx_) * 1e-6;
    }
    
    std::cout << "=== END UPDATE ===" << std::endl;
}

// ============================================================================
// LANDMARK MANAGEMENT
// ============================================================================

void UKF::setLandmarks(const std::map<int, std::pair<double, double>>& landmarks) {
    landmarks_ = landmarks;
}

bool UKF::hasLandmark(int id) const {
    return landmarks_.find(id) != landmarks_.end();
}
