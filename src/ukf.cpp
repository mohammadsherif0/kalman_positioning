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
    
    // 2. Initialize state covariance P as identity matrix
    P_ = Eigen::MatrixXd::Identity(nx_, nx_) * 10.0;  // Moderate initial uncertainty
    
    // 3. Set process noise covariance Q
    Q_ = Eigen::MatrixXd::Zero(nx_, nx_);
    Q_(0, 0) = process_noise_xy;      // x
    Q_(1, 1) = process_noise_xy;      // y
    Q_(2, 2) = process_noise_theta;   // theta
    Q_(3, 3) = 0.1;                   // vx (small)
    Q_(4, 4) = 0.1;                   // vy (small)
    
    // 4. Set measurement noise covariance R
    R_ = Eigen::MatrixXd::Identity(nz_, nz_) * measurement_noise_xy;
    
    // 5. Calculate UKF parameters for ALL POSITIVE weights
    // Using kappa = 1 ensures numerical stability
    double kappa = 1.0;
    lambda_ = ALPHA * ALPHA * (nx_ + kappa) - nx_;  // alpha=1, so lambda = 1
    gamma_ = std::sqrt(nx_ + lambda_);               // gamma = sqrt(6) = 2.45
    
    // 6. Calculate weights (simplified for stability)
    Wm_.resize(2 * nx_ + 1);
    Wc_.resize(2 * nx_ + 1);
    
    // Use equal weights for maximum stability
    double weight_0 = 1.0 / (2.0 * nx_ + 1.0);
    double weight_i = 1.0 / (2.0 * nx_ + 1.0);
    
    Wm_[0] = weight_0;
    Wc_[0] = weight_0;
    
    for (int i = 1; i < 2 * nx_ + 1; i++) {
        Wm_[i] = weight_i;
        Wc_[i] = weight_i;
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
    
    // 2. Compute matrix square root using eigenvalue decomposition
    // P = V * D * V^T, so sqrt(P) = V * sqrt(D)
    // Then scale by sqrt(2*n+1) for equal weights
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    Eigen::MatrixXd eigenvectors = solver.eigenvectors();
    
    // Ensure positive eigenvalues
    for (int i = 0; i < eigenvalues.size(); i++) {
        if (eigenvalues(i) < 1e-10) eigenvalues(i) = 1e-10;
    }
    
    // Compute sqrt(P) = V * sqrt(D), then scale for equal weights
    double scale = std::sqrt(2.0 * n + 1.0);
    Eigen::MatrixXd sqrt_matrix = eigenvectors * eigenvalues.cwiseSqrt().asDiagonal() * scale;
    
    // 3. Generate 2*n sigma points
    for (int i = 0; i < n; i++) {
        sigma_points.push_back(mean + sqrt_matrix.col(i));  // Positive direction
    }
    for (int i = 0; i < n; i++) {
        sigma_points.push_back(mean - sqrt_matrix.col(i));  // Negative direction
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
    
    if (landmarks_.find(landmark_id) == landmarks_.end()) {
        return Eigen::Vector2d::Zero();
    }
    
    // Get landmark position
    auto landmark = landmarks_[landmark_id];
    double lx = landmark.first;
    double ly = landmark.second;
    
    // Get robot state
    double rx = state(0);
    double ry = state(1);
    double theta = state(2);
    
    // Calculate relative position in world frame
    double dx_world = lx - rx;
    double dy_world = ly - ry;
    
    // Transform to robot frame
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    
    Eigen::Vector2d measurement;
    measurement(0) = cos_theta * dx_world + sin_theta * dy_world;   // rel_x
    measurement(1) = -sin_theta * dx_world + cos_theta * dy_world;  // rel_y
    
    return measurement;
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
    
    // Task A6: Update Step
    // ========================================================================
    
    // Process each landmark observation
    for (const auto& obs : landmark_observations) {
        int landmark_id = std::get<0>(obs);
        double obs_x = std::get<1>(obs);
        double obs_y = std::get<2>(obs);
        
        // Skip if landmark not known
        if (!hasLandmark(landmark_id)) {
            continue;
        }
        
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
            x_diff(2) = normalizeAngle(x_diff(2));  // Normalize angle
            P_xz += Wc_[i] * (x_diff * z_diff.transpose());
        }
        
        // Add measurement noise
        P_zz += R_;
        
        // 5. Compute Kalman gain
        Eigen::MatrixXd K = P_xz * P_zz.inverse();
        
        // 6. Update state with innovation
        Eigen::Vector2d z_obs(obs_x, obs_y);
        Eigen::Vector2d innovation = z_obs - z_mean;
        x_ = x_ + K * innovation;
        x_(2) = normalizeAngle(x_(2));  // Normalize angle
        
        // 7. Update covariance
        P_ = P_ - K * P_zz * K.transpose();
        
        // Ensure P remains symmetric
        P_ = (P_ + P_.transpose()) / 2.0;
    }
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
