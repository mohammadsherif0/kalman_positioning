#!/usr/bin/env python3
"""
Positioning Node for UKF-based Robot Localization

This node subscribes to:
  - /robot_noisy: Noisy odometry (dead-reckoning)
  - /landmarks_observed: Noisy landmark observations

And publishes to:
  - /robot_estimated_odometry: Estimated pose and velocity from UKF filter
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import Quaternion, Twist, Pose, PoseWithCovariance, TwistWithCovariance
import numpy as np
import math
import os

from .ukf import UKF
from .landmark_manager import LandmarkManager


def euler_from_quaternion(quat):
    """Convert quaternion (x, y, z, w) to euler angles (roll, pitch, yaw)."""
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def quaternion_from_euler(roll, pitch, yaw):
    """Convert euler angles (roll, pitch, yaw) to quaternion (x, y, z, w)."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return [x, y, z, w]


class PositioningNode(Node):
    """ROS2 node for UKF-based robot localization."""
    
    def __init__(self):
        super().__init__('kalman_positioning_node')
        
        self.get_logger().info('Initializing Kalman Positioning Node')
        
        # Declare parameters
        self.declare_parameter('landmarks_csv_path', 'landmarks.csv')
        self.declare_parameter('process_noise_xy', 0.001)
        self.declare_parameter('process_noise_theta', 0.001)
        self.declare_parameter('measurement_noise_xy', 0.001)
        self.declare_parameter('observation_radius', 10.0)
        self.declare_parameter('decomposition', 'svd')  # 'cholesky' or 'svd'
        
        # Get parameters
        landmarks_file = self.get_parameter('landmarks_csv_path').value
        process_noise_xy = self.get_parameter('process_noise_xy').value
        process_noise_theta = self.get_parameter('process_noise_theta').value
        measurement_noise_xy = self.get_parameter('measurement_noise_xy').value
        self.observation_radius = self.get_parameter('observation_radius').value
        decomposition = self.get_parameter('decomposition').value
        
        # Load landmarks
        self.landmark_manager = LandmarkManager()
        
        # Try to find landmarks file in workspace
        if os.path.isabs(landmarks_file):
            landmarks_path = landmarks_file
        else:
            # First try relative to package root
            pkg_root = os.path.dirname(os.path.dirname(__file__))
            landmarks_path = os.path.join(pkg_root, landmarks_file)
            if not os.path.exists(landmarks_path):
                # Try relative to workspace root (one level up)
                workspace_root = os.path.dirname(pkg_root)
                alt_path = os.path.join(workspace_root, landmarks_file)
                if os.path.exists(alt_path):
                    landmarks_path = alt_path
                else:
                    # Try installed share directory
                    from ament_index_python.packages import get_package_share_directory
                    try:
                        pkg_share = get_package_share_directory('kalman_positioning')
                        share_path = os.path.join(pkg_share, landmarks_file)
                        if os.path.exists(share_path):
                            landmarks_path = share_path
                    except Exception:
                        pass
        
        if not self.landmark_manager.load_from_csv(landmarks_path):
            self.get_logger().error(f'Failed to load landmarks from {landmarks_path}')
            raise RuntimeError('Failed to load landmarks')
        
        # Initialize UKF
        num_landmarks = self.landmark_manager.get_num_landmarks()
        self.ukf = UKF(
            process_noise_xy=process_noise_xy,
            process_noise_theta=process_noise_theta,
            measurement_noise_xy=measurement_noise_xy,
            num_landmarks=num_landmarks,
            decomposition=decomposition
        )
        
        # Set landmarks in UKF
        self.ukf.set_landmarks(self.landmark_manager.get_landmarks())
        
        # State tracking
        self.last_odom_msg = None
        self.last_time = None
        self.initialized = False
        
        # Create subscribers
        self.odometry_sub = self.create_subscription(
            Odometry,
            '/robot_noisy',
            self.odometry_callback,
            10
        )
        self.get_logger().info('Subscribed to /robot_noisy')
        
        self.landmarks_obs_sub = self.create_subscription(
            PointCloud2,
            '/landmarks_observed',
            self.landmarks_observed_callback,
            10
        )
        self.get_logger().info('Subscribed to /landmarks_observed')
        
        # Create publisher
        self.estimated_odom_pub = self.create_publisher(
            Odometry,
            '/robot_estimated_odometry',
            10
        )
        self.get_logger().info('Publishing to /robot_estimated_odometry')
        
        self.get_logger().info('Kalman Positioning Node initialized successfully')
        self.get_logger().info(f'Parameters:')
        self.get_logger().info(f'  Landmarks: {landmarks_path}')
        self.get_logger().info(f'  Process noise (xy, theta): {process_noise_xy}, {process_noise_theta}')
        self.get_logger().info(f'  Measurement noise: {measurement_noise_xy}')
        self.get_logger().info(f'  Observation radius: {self.observation_radius}m')
        self.get_logger().info(f'  Decomposition: {decomposition}')
    
    def odometry_callback(self, msg: Odometry):
        """
        Callback for noisy odometry measurements.
        
        This implements the prediction step of the UKF using odometry data.
        """
        # Extract position and orientation
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = self.quaternion_to_yaw(msg.pose.pose.orientation)
        
        # Get current time
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Initialize filter on first message
        if not self.initialized:
            self.ukf.x = np.array([x, y, theta, 0.0, 0.0])
            self.last_odom_msg = msg
            self.last_time = current_time
            self.initialized = True
            self.get_logger().info(f'Filter initialized at position ({x:.2f}, {y:.2f}), theta={theta:.2f}')
            
            # Publish initial estimate
            self.publish_estimated_odometry(msg.header.stamp)
            return
        
        # Calculate odometry delta (motion since last update)
        if self.last_odom_msg is not None:
            last_x = self.last_odom_msg.pose.pose.position.x
            last_y = self.last_odom_msg.pose.pose.position.y
            last_theta = self.quaternion_to_yaw(self.last_odom_msg.pose.pose.orientation)
            
            dx = x - last_x
            dy = y - last_y
            dtheta = self.normalize_angle(theta - last_theta)
            dt = current_time - self.last_time
            
            # Prediction step
            if dt > 0:
                self.ukf.predict(dt, dx, dy, dtheta)
                self.get_logger().debug(
                    f'Prediction: dt={dt:.3f}, dx={dx:.3f}, dy={dy:.3f}, dtheta={dtheta:.3f}'
                )
        
        # Update state tracking
        self.last_odom_msg = msg
        self.last_time = current_time
        
        # Publish estimated odometry
        self.publish_estimated_odometry(msg.header.stamp)
    
    def landmarks_observed_callback(self, msg: PointCloud2):
        """
        Callback for landmark observations.
        
        This implements the measurement update step of the UKF using landmark observations.
        """
        if not self.initialized:
            return
        
        try:
            # Parse PointCloud2 data; incoming points appear to be in map frame.
            # Convert to robot frame using the latest odometry pose (not the UKF
            # state) so the measurement comes from sensor data and can correct
            # the filter state.
            if self.last_odom_msg is None:
                return

            odom_theta = self.quaternion_to_yaw(self.last_odom_msg.pose.pose.orientation)
            odom_x = self.last_odom_msg.pose.pose.position.x
            odom_y = self.last_odom_msg.pose.pose.position.y
            c, s = math.cos(odom_theta), math.sin(odom_theta)

            observations = []
            
            for point in point_cloud2.read_points(msg, field_names=('x', 'y', 'id'), skip_nans=True):
                landmark_id = int(point[2])
                obs_x = float(point[0])
                obs_y = float(point[1])
                
                if self.ukf.has_landmark(landmark_id):
                    dx = obs_x - odom_x
                    dy = obs_y - odom_y
                    rel_x = c * dx + s * dy
                    rel_y = -s * dx + c * dy
                    observations.append((landmark_id, rel_x, rel_y))
                    self.get_logger().debug(
                        f'Landmark {landmark_id} map=({obs_x:.2f},{obs_y:.2f}) rel=({rel_x:.2f},{rel_y:.2f})'
                    )
            
            if observations:
                # Update step with all observations
                self.ukf.update(observations)
                self.get_logger().debug(f'Updated with {len(observations)} landmark observations')
                
                # Publish updated estimate
                self.publish_estimated_odometry(msg.header.stamp)
            else:
                self.get_logger().debug('No valid landmark observations')
                
        except Exception as e:
            self.get_logger().warn(f'Failed to parse landmark observations: {e}')
    
    def publish_estimated_odometry(self, timestamp):
        """Publish the estimated odometry from the UKF."""
        # Get state estimate
        x, y = self.ukf.get_position()
        theta = self.ukf.get_orientation()
        vx, vy = self.ukf.get_velocity()
        cov = self.ukf.get_covariance()
        
        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp
        odom_msg.header.frame_id = 'map'
        odom_msg.child_frame_id = 'robot_estimated'
        
        # Set position
        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.position.z = 0.0
        
        # Set orientation (convert yaw to quaternion)
        q = quaternion_from_euler(0, 0, theta)
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]
        
        # Set velocity
        odom_msg.twist.twist.linear.x = vx
        odom_msg.twist.twist.linear.y = vy
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.z = 0.0
        
        # Set covariance (6x6 for pose, 6x6 for twist)
        # Map 5x5 state covariance to 6x6 pose covariance
        pose_cov = np.zeros((6, 6))
        pose_cov[0, 0] = cov[0, 0]  # x
        pose_cov[1, 1] = cov[1, 1]  # y
        pose_cov[5, 5] = cov[2, 2]  # theta (yaw)
        odom_msg.pose.covariance = pose_cov.flatten().tolist()
        
        twist_cov = np.zeros((6, 6))
        twist_cov[0, 0] = cov[3, 3]  # vx
        twist_cov[1, 1] = cov[4, 4]  # vy
        odom_msg.twist.covariance = twist_cov.flatten().tolist()
        
        # Publish
        self.estimated_odom_pub.publish(odom_msg)
        
        self.get_logger().debug(
            f'Published estimate: x={x:.2f}, y={y:.2f}, theta={theta:.2f}'
        )
    
    def quaternion_to_yaw(self, q: Quaternion) -> float:
        """Convert quaternion to yaw angle."""
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = PositioningNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

