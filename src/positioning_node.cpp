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
#include <string>
#include <tuple>
#include <vector>
#include <array>

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
    PositioningNode() : Node("kalman_positioning_node") {
        RCLCPP_INFO(this->get_logger(), "Initializing Kalman Positioning Node");
        
        // Declare parameters
        std::string landmarks_csv_path = this->declare_parameter<std::string>("landmarks_csv_path", "landmarks.csv");
        process_noise_xy_ = this->declare_parameter<double>("process_noise_xy", 1e-4);
        process_noise_theta_ = this->declare_parameter<double>("process_noise_theta", 1e-4);
        measurement_noise_xy_ = this->declare_parameter<double>("measurement_noise_xy", 0.01);
        observation_radius_ = this->declare_parameter<double>("observation_radius", 5.0);
        
        if (!landmark_manager_.loadFromCSV(landmarks_csv_path)) {
            RCLCPP_WARN(this->get_logger(), "Failed to load landmarks from %s", landmarks_csv_path.c_str());
        } else {
            RCLCPP_INFO(this->get_logger(), "Loaded %zu landmarks", landmark_manager_.getNumLandmarks());
        }
        
        ukf_ = std::make_unique<UKF>(
            process_noise_xy_,
            process_noise_theta_,
            measurement_noise_xy_,
            static_cast<int>(landmark_manager_.getNumLandmarks()));
        ukf_->setLandmarks(landmark_manager_.getLandmarks());
        
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
    // PLACEHOLDER: KALMAN FILTER STATE
    // ============================================================================
    // Students should implement a proper Kalman filter (e.g., UKF, EKF) 
    // with the following state:
    //   - Position: x, y (m)
    //   - Orientation: theta (rad)
    //   - Velocity: vx, vy (m/s)
    // And maintain:
    //   - State covariance matrix
    //   - Process noise covariance
    //   - Measurement noise covariance
    LandmarkManager landmark_manager_;
    std::unique_ptr<UKF> ukf_;
    double process_noise_xy_{1e-4};
    double process_noise_theta_{1e-4};
    double measurement_noise_xy_{0.01};
    double observation_radius_{5.0};
    
    bool has_previous_odom_{false};
    double last_x_{0.0};
    double last_y_{0.0};
    double last_theta_{0.0};
    rclcpp::Time last_stamp_;
    
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
        RCLCPP_DEBUG(this->get_logger(), 
            "Odometry received: x=%.3f, y=%.3f", 
            msg->pose.pose.position.x, msg->pose.pose.position.y);
        
        // STUDENT ASSIGNMENT STARTS HERE
        // ========================================================================
        
        // Placeholder: Extract and log the data
        double x = msg->pose.pose.position.x;
        double y = msg->pose.pose.position.y;
        double theta = quaternionToYaw(msg->pose.pose.orientation);
        double vx = msg->twist.twist.linear.x;
        double vy = msg->twist.twist.linear.y;
        
        RCLCPP_DEBUG(this->get_logger(), 
            "Parsed: x=%.3f, y=%.3f, theta=%.3f, vx=%.3f, vy=%.3f",
            x, y, theta, vx, vy);
        
        double dt = 0.0;
        double dx = 0.0;
        double dy = 0.0;
        double dtheta = 0.0;
        
        rclcpp::Time current_stamp(msg->header.stamp);
        if (has_previous_odom_) {
            dt = (current_stamp - last_stamp_).seconds();
            dx = x - last_x_;
            dy = y - last_y_;
            dtheta = normalizeAngle(theta - last_theta_);
            
            if (ukf_) {
                ukf_->predict(dt, dx, dy, dtheta);
            }
        }
        
        last_x_ = x;
        last_y_ = y;
        last_theta_ = theta;
        last_stamp_ = current_stamp;
        has_previous_odom_ = true;
        
        publishEstimatedOdometry(msg->header.stamp, *msg);
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
        // STUDENT ASSIGNMENT STARTS HERE
        // ========================================================================
        
        RCLCPP_DEBUG(this->get_logger(), 
            "Landmark observation received with %u points", msg->width);
        
        // Placeholder: Parse and log the observations
        try {
            sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
            sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
            sensor_msgs::PointCloud2ConstIterator<uint32_t> iter_id(*msg, "id");
            
            int count = 0;
            std::vector<std::tuple<int, double, double, double>> observations;
            for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_id) {
                int landmark_id = static_cast<int>(*iter_id);
                float obs_x = static_cast<float>(*iter_x);
                float obs_y = static_cast<float>(*iter_y);
                
                RCLCPP_DEBUG(this->get_logger(),
                    "Landmark %d observed at (%.3f, %.3f)",
                    landmark_id, obs_x, obs_y);
                
                observations.emplace_back(
                    landmark_id,
                    static_cast<double>(obs_x),
                    static_cast<double>(obs_y),
                    measurement_noise_xy_);
                count++;
            }
            
            RCLCPP_DEBUG(this->get_logger(), 
                "Processed %d landmark observations", count);
            
            if (!observations.empty() && ukf_) {
                ukf_->update(observations);
                publishEstimatedOdometry(msg->header.stamp, nav_msgs::msg::Odometry());
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
     * @brief Publish estimated odometry message
     * @param timestamp Message timestamp
     * @param odom_msg Odometry message to publish
     */
    void publishEstimatedOdometry(const rclcpp::Time& timestamp, 
                                  const nav_msgs::msg::Odometry& odom_msg) {
        nav_msgs::msg::Odometry estimated_odom;
        estimated_odom.header.stamp = timestamp;
        estimated_odom.header.frame_id = "map";
        estimated_odom.child_frame_id = "robot_estimated";
        
        if (ukf_) {
            Eigen::VectorXd state = ukf_->getState();
            Eigen::MatrixXd cov = ukf_->getCovariance();
            
            estimated_odom.pose.pose.position.x = state(0);
            estimated_odom.pose.pose.position.y = state(1);
            estimated_odom.pose.pose.position.z = 0.0;
            
            tf2::Quaternion q;
            q.setRPY(0.0, 0.0, state(2));
            estimated_odom.pose.pose.orientation = tf2::toMsg(q);
            
            estimated_odom.twist.twist.linear.x = state(3);
            estimated_odom.twist.twist.linear.y = state(4);
            estimated_odom.twist.twist.linear.z = 0.0;
            estimated_odom.twist.twist.angular.x = 0.0;
            estimated_odom.twist.twist.angular.y = 0.0;
            estimated_odom.twist.twist.angular.z = 0.0;
            
            std::array<double, 36> pose_covariance{};
            pose_covariance[0] = cov(0, 0);
            pose_covariance[1] = cov(0, 1);
            pose_covariance[6] = cov(1, 0);
            pose_covariance[7] = cov(1, 1);
            pose_covariance[35] = cov(2, 2);
            estimated_odom.pose.covariance = pose_covariance;
            
            std::array<double, 36> twist_covariance{};
            twist_covariance[0] = cov(3, 3);
            twist_covariance[1] = cov(3, 4);
            twist_covariance[6] = cov(4, 3);
            twist_covariance[7] = cov(4, 4);
            twist_covariance[35] = cov(2, 2);
            estimated_odom.twist.covariance = twist_covariance;
        } else {
            // Fallback to raw message only if filter unavailable
            estimated_odom = odom_msg;
            estimated_odom.header.stamp = timestamp;
            estimated_odom.header.frame_id = "map";
            estimated_odom.child_frame_id = "robot_estimated";
        }
        
        estimated_odom_pub_->publish(estimated_odom);
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PositioningNode>());
    rclcpp::shutdown();
    return 0;
}
