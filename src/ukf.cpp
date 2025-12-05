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
    
    // STUDENT IMPLEMENTATION STARTS HERE
    // ========================================================================
    
    lambda_ = ALPHA * ALPHA * (nx_ + KAPPA) - nx_;
    gamma_ = std::sqrt(nx_ + lambda_);
    
    x_ = Eigen::VectorXd::Zero(nx_);
    P_ = Eigen::MatrixXd::Identity(nx_, nx_);
    
    Q_ = Eigen::MatrixXd::Zero(nx_, nx_);
    Q_(0, 0) = process_noise_xy;
    Q_(1, 1) = process_noise_xy;
    Q_(2, 2) = process_noise_theta;
    
    R_ = Eigen::MatrixXd::Zero(nz_, nz_);
    R_(0, 0) = measurement_noise_xy;
    R_(1, 1) = measurement_noise_xy;
    
    const int sigma_count = 2 * nx_ + 1;
    Wm_.resize(sigma_count, 0.0);
    Wc_.resize(sigma_count, 0.0);
    
    Wm_[0] = lambda_ / (nx_ + lambda_);
    Wc_[0] = lambda_ / (nx_ + lambda_) + (1 - ALPHA * ALPHA + BETA);
    for (int i = 1; i < sigma_count; ++i) {
        Wm_[i] = 1.0 / (2.0 * (nx_ + lambda_));
        Wc_[i] = 1.0 / (2.0 * (nx_ + lambda_));
    }
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
    // STUDENT IMPLEMENTATION STARTS HERE
    // ========================================================================
    
    std::vector<Eigen::VectorXd> sigma_points;
    sigma_points.reserve(2 * nx_ + 1);
    
    Eigen::MatrixXd L = cov.llt().matrixL();
    
    sigma_points.push_back(mean);
    for (int i = 0; i < nx_; ++i) {
        sigma_points.push_back(mean + gamma_ * L.col(i));
        sigma_points.push_back(mean - gamma_ * L.col(i));
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
    // STUDENT IMPLEMENTATION STARTS HERE
    // ========================================================================
    
    Eigen::VectorXd new_state = state;
    
    new_state(0) += dx;
    new_state(1) += dy;
    new_state(2) = normalizeAngle(new_state(2) + dtheta);
    
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
    // STUDENT IMPLEMENTATION STARTS HERE
    // ========================================================================
    
    if (landmarks_.find(landmark_id) == landmarks_.end()) {
        return Eigen::Vector2d::Zero();
    }
    
    const auto& landmark = landmarks_.at(landmark_id);
    Eigen::Vector2d relative;
    relative << landmark.first - state(0), landmark.second - state(1);
    return relative;
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
    // STUDENT IMPLEMENTATION STARTS HERE
    // ========================================================================
    
    auto sigma_points = generateSigmaPoints(x_, P_);
    
    std::vector<Eigen::VectorXd> sigma_pred;
    sigma_pred.reserve(sigma_points.size());
    for (const auto& sp : sigma_points) {
        sigma_pred.push_back(processModel(sp, dt, dx, dy, dtheta));
    }
    
    Eigen::VectorXd x_pred = Eigen::VectorXd::Zero(nx_);
    for (size_t i = 0; i < sigma_pred.size(); ++i) {
        x_pred += Wm_[i] * sigma_pred[i];
    }
    x_pred(2) = normalizeAngle(x_pred(2));
    
    Eigen::MatrixXd P_pred = Eigen::MatrixXd::Zero(nx_, nx_);
    for (size_t i = 0; i < sigma_pred.size(); ++i) {
        Eigen::VectorXd diff = sigma_pred[i] - x_pred;
        diff(2) = normalizeAngle(diff(2));
        P_pred += Wc_[i] * diff * diff.transpose();
    }
    P_pred += Q_;
    
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
    
    // STUDENT IMPLEMENTATION STARTS HERE
    // ========================================================================
    
    for (const auto& obs : landmark_observations) {
        int landmark_id;
        double obs_x, obs_y, obs_noise;
        std::tie(landmark_id, obs_x, obs_y, obs_noise) = obs;
        
        if (!hasLandmark(landmark_id)) {
            continue;
        }
        
        auto sigma_points = generateSigmaPoints(x_, P_);
        std::vector<Eigen::Vector2d> z_sigma;
        z_sigma.reserve(sigma_points.size());
        for (const auto& sp : sigma_points) {
            z_sigma.push_back(measurementModel(sp, landmark_id));
        }
        
        Eigen::Vector2d z_pred = Eigen::Vector2d::Zero();
        for (size_t i = 0; i < z_sigma.size(); ++i) {
            z_pred += Wm_[i] * z_sigma[i];
        }
        
        Eigen::MatrixXd P_zz = Eigen::MatrixXd::Zero(nz_, nz_);
        Eigen::MatrixXd P_xz = Eigen::MatrixXd::Zero(nx_, nz_);
        
        for (size_t i = 0; i < z_sigma.size(); ++i) {
            Eigen::Vector2d z_diff = z_sigma[i] - z_pred;
            Eigen::VectorXd x_diff = sigma_points[i] - x_;
            x_diff(2) = normalizeAngle(x_diff(2));
            
            P_zz += Wc_[i] * z_diff * z_diff.transpose();
            P_xz += Wc_[i] * x_diff * z_diff.transpose();
        }
        
        Eigen::Matrix2d R_obs = R_;
        if (obs_noise > 0.0) {
            R_obs(0, 0) += obs_noise;
            R_obs(1, 1) += obs_noise;
        }
        P_zz += R_obs;
        
        Eigen::MatrixXd K = P_xz * P_zz.inverse();
        Eigen::Vector2d z_meas;
        z_meas << obs_x, obs_y;
        Eigen::Vector2d innovation = z_meas - z_pred;
        
        x_ += K * innovation;
        x_(2) = normalizeAngle(x_(2));
        P_ -= K * P_zz * K.transpose();
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
