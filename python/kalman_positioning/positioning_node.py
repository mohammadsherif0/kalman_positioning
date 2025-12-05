#!/usr/bin/env python3
"""
ROS2 Positioning Node - Python UKF Implementation

This node integrates the Python UKF with ROS2 for visualization in RViz2.

Subscribes to:
  - /robot_noisy: Noisy odometry (nav_msgs/Odometry)
  - /landmarks_observed: Landmark observations (sensor_msgs/PointCloud2)

Publishes to:
  - /robot_estimated_odometry: UKF estimated odometry (nav_msgs/Odometry)
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import Quaternion, TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import os

# Import from same package (relative import)
from . import UKF, LandmarkManager


class PositioningNode(Node):
    """ROS2 node for UKF-based robot localization"""
    
    def __init__(self):
        super().__init__('python_positioning_node')
        
        # Declare parameters
        self.declare_parameter('landmarks_csv_path', '')
        self.declare_parameter('process_noise_xy', 1e-4)
        self.declare_parameter('process_noise_theta', 1e-4)
        self.declare_parameter('measurement_noise_xy', 0.01)
        self.declare_parameter('observation_radius', 5.0)
        
        # Get parameters
        landmarks_csv = self.get_parameter('landmarks_csv_path').value
        process_noise_xy = self.get_parameter('process_noise_xy').value
        process_noise_theta = self.get_parameter('process_noise_theta').value
        measurement_noise_xy = self.get_parameter('measurement_noise_xy').value
        self.observation_radius = self.get_parameter('observation_radius').value
        
        self.get_logger().info('=' * 70)
        self.get_logger().info('Python UKF Positioning Node Initializing')
        self.get_logger().info('=' * 70)
        
        # Load landmarks
        self.landmark_manager = LandmarkManager()
        if landmarks_csv and os.path.exists(landmarks_csv):
            if self.landmark_manager.load_from_csv(landmarks_csv):
                self.get_logger().info(f'Loaded {self.landmark_manager.get_num_landmarks()} landmarks from {landmarks_csv}')
            else:
                self.get_logger().warn(f'Failed to load landmarks from {landmarks_csv}')
        else:
            # Try default location (parent of python folder)
            default_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'landmarks.csv')
            if os.path.exists(default_csv):
                if self.landmark_manager.load_from_csv(default_csv):
                    self.get_logger().info(f'Loaded {self.landmark_manager.get_num_landmarks()} landmarks from {default_csv}')
            else:
                self.get_logger().error(f'No landmarks file found! Tried: {default_csv}')
        
        # Initialize UKF
        self.ukf = UKF(
            process_noise_xy=process_noise_xy,
            process_noise_theta=process_noise_theta,
            measurement_noise_xy=measurement_noise_xy,
            num_landmarks=self.landmark_manager.get_num_landmarks()
        )
        self.ukf.set_landmarks(self.landmark_manager.get_landmarks())
        
        self.get_logger().info(f'UKF Parameters:')
        self.get_logger().info(f'  Process Noise (XY): {process_noise_xy}')
        self.get_logger().info(f'  Process Noise (Theta): {process_noise_theta}')
        self.get_logger().info(f'  Measurement Noise: {measurement_noise_xy}')
        self.get_logger().info(f'  Observation Radius: {self.observation_radius}m')
        
        # Previous odometry for calculating motion
        self.prev_odom = None
        self.prev_time = None
        
        # Create subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/robot_noisy',
            self.odometry_callback,
            10
        )
        
        self.landmarks_sub = self.create_subscription(
            PointCloud2,
            '/landmarks_observed',
            self.landmarks_callback,
            10
        )
        
        # Create publisher
        self.estimated_odom_pub = self.create_publisher(
            Odometry,
            '/robot_estimated_odometry',
            10
        )
        
        # TF broadcaster (optional)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.get_logger().info('Subscribed to /robot_noisy and /landmarks_observed')
        self.get_logger().info('Publishing to /robot_estimated_odometry')
        self.get_logger().info('=' * 70)
        self.get_logger().info('Python UKF Node Ready!')
        self.get_logger().info('=' * 70)
    
    def quaternion_to_yaw(self, q):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return yaw
    
    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion"""
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = np.sin(yaw / 2.0)
        q.w = np.cos(yaw / 2.0)
        return q
    
    def odometry_callback(self, msg):
        """Handle odometry messages - UKF prediction step"""
        current_time = self.get_clock().now()
        
        # Extract current odometry
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = self.quaternion_to_yaw(msg.pose.pose.orientation)
        
        # Calculate motion since last odometry
        if self.prev_odom is not None and self.prev_time is not None:
            # Calculate displacement
            dx = x - self.prev_odom[0]
            dy = y - self.prev_odom[1]
            dtheta = theta - self.prev_odom[2]
            
            # Normalize angle difference
            while dtheta > np.pi:
                dtheta -= 2.0 * np.pi
            while dtheta < -np.pi:
                dtheta += 2.0 * np.pi
            
            # Calculate time step
            dt = (current_time - self.prev_time).nanoseconds / 1e9
            
            if dt > 0:
                # UKF Prediction step
                self.ukf.predict(dt, dx, dy, dtheta)
                
                # Publish estimated odometry after prediction
                self.publish_estimated_odometry(current_time)
                
                self.get_logger().debug(
                    f'Predict: dt={dt:.3f}s, dx={dx:.3f}, dy={dy:.3f}, dtheta={dtheta:.3f}'
                )
        
        # Store current odometry for next iteration
        self.prev_odom = (x, y, theta)
        self.prev_time = current_time
    
    def landmarks_callback(self, msg):
        """Handle landmark observations - UKF update step"""
        try:
            # Parse PointCloud2 data
            observations = []
            
            for point in point_cloud2.read_points(msg, field_names=("x", "y", "id"), skip_nans=True):
                landmark_id = int(point[2])
                obs_x = float(point[0])
                obs_y = float(point[1])
                
                # Check if landmark is known
                if self.ukf.has_landmark(landmark_id):
                    # Add observation (id, x, y, noise_cov)
                    observations.append((landmark_id, obs_x, obs_y, 0.01))
                    
                    self.get_logger().debug(
                        f'Landmark {landmark_id} observed at ({obs_x:.3f}, {obs_y:.3f})'
                    )
            
            if observations:
                # UKF Update step
                self.ukf.update(observations)
                
                # Publish estimated odometry after update
                current_time = self.get_clock().now()
                self.publish_estimated_odometry(current_time)
                
                self.get_logger().debug(f'Updated with {len(observations)} landmark observations')
            
        except Exception as e:
            self.get_logger().warn(f'Failed to process landmarks: {str(e)}')
    
    def publish_estimated_odometry(self, timestamp):
        """Publish the UKF estimated odometry"""
        # Get UKF state
        state = self.ukf.get_state()
        covariance = self.ukf.get_covariance()
        
        x, y, theta, vx, vy = state
        
        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp.to_msg()
        odom_msg.header.frame_id = 'map'
        odom_msg.child_frame_id = 'robot_estimated'
        
        # Set position
        odom_msg.pose.pose.position.x = float(x)
        odom_msg.pose.pose.position.y = float(y)
        odom_msg.pose.pose.position.z = 0.0
        
        # Set orientation (from theta)
        odom_msg.pose.pose.orientation = self.yaw_to_quaternion(theta)
        
        # Set velocity
        odom_msg.twist.twist.linear.x = float(vx)
        odom_msg.twist.twist.linear.y = float(vy)
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.z = 0.0
        
        # Set covariance (6x6 pose covariance)
        pose_cov = np.zeros((6, 6))
        pose_cov[0, 0] = covariance[0, 0]  # x-x
        pose_cov[1, 1] = covariance[1, 1]  # y-y
        pose_cov[5, 5] = covariance[2, 2]  # theta-theta
        pose_cov[0, 1] = covariance[0, 1]  # x-y
        pose_cov[1, 0] = covariance[1, 0]  # y-x
        odom_msg.pose.covariance = pose_cov.flatten().tolist()
        
        # Set twist covariance
        twist_cov = np.zeros((6, 6))
        twist_cov[0, 0] = covariance[3, 3]  # vx-vx
        twist_cov[1, 1] = covariance[4, 4]  # vy-vy
        odom_msg.twist.covariance = twist_cov.flatten().tolist()
        
        # Publish
        self.estimated_odom_pub.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = PositioningNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

