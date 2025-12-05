#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include "kalman_positioning/ukf.hpp"
#include "kalman_positioning/landmark_manager.hpp"

#include <memory>
#include <cmath>

/**
 * @brief Positioning node for UKF-based robot localization (Student Assignment)
 * 
 * This node subscribes to:
 *   - /robot_noisy: Noisy odometry (dead-reckoning)
 *   - /landmarks_observed: Noisy landmark observations
 * 
 * And publishes to:
 *   - /robot_estimated_odometry: Estimated pose and velocity from filter
 * 
 * STUDENT ASSIGNMENT:
 * Implement the Kalman filter logic to fuse odometry and landmark observations
 * to estimate the robot's true position.
 */
class PositioningNode : public rclcpp::Node {
public:
    PositioningNode() : Node("kalman_positioning_node"), initialized_(false) {
        RCLCPP_INFO(this->get_logger(), "Initializing Kalman Positioning Node");
        
        // Declare parameters
        this->declare_parameter("landmarks_csv_path", "../landmarks.csv");
        this->declare_parameter("process_noise_xy", 0.01);
        this->declare_parameter("process_noise_theta", 0.01);
        this->declare_parameter("measurement_noise_xy", 0.01);
        
        // Get parameters
        std::string landmarks_csv = this->get_parameter("landmarks_csv_path").as_string();
        double process_noise_xy = this->get_parameter("process_noise_xy").as_double();
        double process_noise_theta = this->get_parameter("process_noise_theta").as_double();
        double measurement_noise_xy = this->get_parameter("measurement_noise_xy").as_double();
        
        // Load landmarks
        landmark_manager_ = std::make_shared<LandmarkManager>();
        if (landmarks_csv.empty()) {
            RCLCPP_WARN(this->get_logger(), "No landmarks CSV path provided!");
        } else if (landmark_manager_->loadFromCSV(landmarks_csv)) {
            RCLCPP_INFO(this->get_logger(), "Loaded %zu landmarks from %s", 
                        landmark_manager_->getNumLandmarks(), landmarks_csv.c_str());
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to load landmarks from %s", landmarks_csv.c_str());
        }
        
        // Initialize UKF
        ukf_ = std::make_shared<UKF>(process_noise_xy, process_noise_theta, 
                                      measurement_noise_xy, 
                                      landmark_manager_->getNumLandmarks());
        ukf_->setLandmarks(landmark_manager_->getLandmarks());
        
        RCLCPP_INFO(this->get_logger(), "UKF initialized with process_noise=%.4f, meas_noise=%.4f",
                    process_noise_xy, measurement_noise_xy);
        
        // Create subscribers
        odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/robot_noisy",
            rclcpp::QoS(10),
            std::bind(&PositioningNode::odometryCallback, this, std::placeholders::_1)
        );
        RCLCPP_INFO(this->get_logger(), "Subscribed to /robot_noisy");
        
        landmarks_obs_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/landmarks_observed",
            rclcpp::QoS(10),
            std::bind(&PositioningNode::landmarksObservedCallback, this, std::placeholders::_1)
        );
        RCLCPP_INFO(this->get_logger(), "Subscribed to /landmarks_observed");
        
        // Create publisher
        estimated_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
            "/robot_estimated_odometry", rclcpp::QoS(10)
        );
        RCLCPP_INFO(this->get_logger(), "Publishing to /robot_estimated_odometry");
        
        RCLCPP_INFO(this->get_logger(), "Kalman Positioning Node initialized successfully");
    }

private:
    // ============================================================================
    // SUBSCRIBERS AND PUBLISHERS
    // ============================================================================
    
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr landmarks_obs_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr estimated_odom_pub_;
    
    // ============================================================================
    // UKF AND LANDMARK MANAGEMENT
    // ============================================================================
    
    std::shared_ptr<UKF> ukf_;
    std::shared_ptr<LandmarkManager> landmark_manager_;
    
    bool initialized_;
    rclcpp::Time last_odom_time_;
    double last_x_, last_y_, last_theta_;
    
    // ============================================================================
    // CALLBACK FUNCTIONS
    // ============================================================================
    
    /**
     * @brief Callback for noisy odometry measurements
     * 
     * STUDENT TODO:
     * 1. Extract position (x, y) and orientation (theta) from the message
     * 2. Update the Kalman filter's prediction step with this odometry
     * 3. Publish the estimated odometry
     */
    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        double x = msg->pose.pose.position.x;
        double y = msg->pose.pose.position.y;
        double theta = quaternionToYaw(msg->pose.pose.orientation);
        
        // Initialize UKF state from first odometry
        if (!initialized_) {
            ukf_->setState(0, x);
            ukf_->setState(1, y);
            ukf_->setState(2, theta);
            last_x_ = x;
            last_y_ = y;
            last_theta_ = theta;
            last_odom_time_ = rclcpp::Time(msg->header.stamp);
            initialized_ = true;
            
            RCLCPP_INFO(this->get_logger(), "UKF initialized from first odometry: (%.2f, %.2f, %.2f)", 
                        x, y, theta);
            publishEstimatedOdometry(rclcpp::Time(msg->header.stamp));
            return;
        }
        
        // Calculate motion since last odometry
        rclcpp::Time current_time = rclcpp::Time(msg->header.stamp);
        double dt = (current_time - last_odom_time_).seconds();
        double dx = x - last_x_;
        double dy = y - last_y_;
        double dtheta = normalizeAngle(theta - last_theta_);
        
        RCLCPP_INFO(this->get_logger(), "ODOM: pos=(%.2f,%.2f,%.2f), delta=(%.3f,%.3f,%.3f)", 
                    x, y, theta, dx, dy, dtheta);
        
        // UKF Prediction Step
        ukf_->predict(dt, dx, dy, dtheta);
        
        Eigen::VectorXd state_after_predict = ukf_->getState();
        RCLCPP_INFO(this->get_logger(), "After PREDICT: (%.2f, %.2f, %.2f)", 
                    state_after_predict(0), state_after_predict(1), state_after_predict(2));
        
        // Update for next callback
        last_x_ = x;
        last_y_ = y;
        last_theta_ = theta;
        last_odom_time_ = current_time;
        
        // Publish estimated odometry
        publishEstimatedOdometry(current_time);
    }
    
    /**
     * @brief Callback for landmark observations
     * 
     * STUDENT TODO:
     * 1. Parse the PointCloud2 data to extract landmark observations
     * 2. Update the Kalman filter's measurement update step with these observations
     * 3. Optionally publish the updated estimated odometry
     */
    void landmarksObservedCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (!initialized_) {
            RCLCPP_DEBUG(this->get_logger(), "Landmark callback: Not initialized yet");
            return;
        }
        
        RCLCPP_DEBUG(this->get_logger(), "Landmark observation received with %u points", msg->width);
        
        // Parse landmark observations
        std::vector<std::tuple<int, double, double, double>> observations;
        
        try {
            sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
            sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
            sensor_msgs::PointCloud2ConstIterator<uint32_t> iter_id(*msg, "id");
            
            for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_id) {
                int landmark_id = static_cast<int>(*iter_id);
                double obs_x = static_cast<double>(*iter_x);
                double obs_y = static_cast<double>(*iter_y);
                
                RCLCPP_DEBUG(this->get_logger(), "  Landmark %d: obs=(%.2f, %.2f)", 
                            landmark_id, obs_x, obs_y);
                
                // Add observation (distance not used in our measurement model)
                observations.push_back(std::make_tuple(landmark_id, obs_x, obs_y, 0.0));
            }
            
            // UKF Update Step
            if (!observations.empty()) {
                ukf_->update(observations);
                RCLCPP_INFO(this->get_logger(), "Updated with %zu landmark observations", 
                            observations.size());
                
                // Log current estimate
                Eigen::VectorXd state = ukf_->getState();
                RCLCPP_INFO(this->get_logger(), "  Estimate: (%.2f, %.2f, %.2f)", 
                           state(0), state(1), state(2));
            } else {
                RCLCPP_WARN(this->get_logger(), "No landmarks observed!");
            }
            
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), 
                "Failed to parse landmark observations: %s", e.what());
        }
    }
    
    // ============================================================================
    // HELPER FUNCTIONS
    // ============================================================================
    
    /**
     * @brief Convert quaternion to yaw angle
     * @param q Quaternion from orientation
     * @return Yaw angle in radians [-pi, pi]
     */
    double quaternionToYaw(const geometry_msgs::msg::Quaternion& q) {
        tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
        return yaw;
    }
    
    /**
     * @brief Normalize angle to [-pi, pi]
     * @param angle Input angle in radians
     * @return Normalized angle in [-pi, pi]
     */
    double normalizeAngle(double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }
    
    /**
     * @brief Publish estimated odometry message from UKF state
     * @param timestamp Message timestamp
     */
    void publishEstimatedOdometry(const rclcpp::Time& timestamp) {
        nav_msgs::msg::Odometry estimated_odom;
        estimated_odom.header.stamp = timestamp;
        estimated_odom.header.frame_id = "map";
        estimated_odom.child_frame_id = "robot_estimated";
        
        // Get state from UKF
        Eigen::VectorXd state = ukf_->getState();
        double x = state(0);
        double y = state(1);
        double theta = state(2);
        double vx = state(3);
        double vy = state(4);
        
        // Set position
        estimated_odom.pose.pose.position.x = x;
        estimated_odom.pose.pose.position.y = y;
        estimated_odom.pose.pose.position.z = 0.0;
        
        // Set orientation (convert yaw to quaternion)
        tf2::Quaternion q;
        q.setRPY(0, 0, theta);
        estimated_odom.pose.pose.orientation.x = q.x();
        estimated_odom.pose.pose.orientation.y = q.y();
        estimated_odom.pose.pose.orientation.z = q.z();
        estimated_odom.pose.pose.orientation.w = q.w();
        
        // Set velocity
        estimated_odom.twist.twist.linear.x = vx;
        estimated_odom.twist.twist.linear.y = vy;
        estimated_odom.twist.twist.linear.z = 0.0;
        estimated_odom.twist.twist.angular.z = 0.0;
        
        estimated_odom_pub_->publish(estimated_odom);
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PositioningNode>());
    rclcpp::shutdown();
    return 0;
}
