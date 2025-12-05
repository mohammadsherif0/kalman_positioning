"""
ROS2 Positioning Node using Python UKF
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import Quaternion
import tf_transformations
import csv
import numpy as np

from .ukf import UKF


class PositioningNode(Node):
    def __init__(self):
        super().__init__('positioning_node')
        
        # Declare parameters
        self.declare_parameter('landmarks_csv_path', '')
        self.declare_parameter('process_noise_xy', 0.001)
        self.declare_parameter('process_noise_theta', 0.001)
        self.declare_parameter('measurement_noise_xy', 0.001)
        self.declare_parameter('observation_radius', 10.0)
        
        # Get parameters
        landmarks_path = self.get_parameter('landmarks_csv_path').value
        process_noise_xy = self.get_parameter('process_noise_xy').value
        process_noise_theta = self.get_parameter('process_noise_theta').value
        measurement_noise_xy = self.get_parameter('measurement_noise_xy').value
        
        # Initialize UKF
        self.ukf = UKF(process_noise_xy, process_noise_theta, measurement_noise_xy)
        
        # Load landmarks
        self.load_landmarks(landmarks_path)
        
        # Initialize tracking variables
        self.initialized = False
        self.last_odom_time = None
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_theta = 0.0
        
        # Create subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/robot_noisy', self.odom_callback, 10)
        self.landmarks_sub = self.create_subscription(
            PointCloud2, '/landmarks_observed', self.landmarks_callback, 10)
        
        # Create publisher
        self.est_odom_pub = self.create_publisher(Odometry, '/robot_estimated_odometry', 10)
        
        self.get_logger().info('Python UKF Positioning Node initialized')
    
    def load_landmarks(self, csv_path):
        """Load landmarks from CSV file."""
        landmarks = {}
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) == 3:
                        landmark_id = int(row[0])
                        x = float(row[1])
                        y = float(row[2])
                        landmarks[landmark_id] = (x, y)
            self.ukf.set_landmarks(landmarks)
            self.get_logger().info(f'Loaded {len(landmarks)} landmarks')
        except Exception as e:
            self.get_logger().error(f'Failed to load landmarks: {e}')
    
    def odom_callback(self, msg):
        """Handle odometry messages - UKF prediction step."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = self.quaternion_to_yaw(msg.pose.pose.orientation)
        
        current_time = self.get_clock().now()
        
        # Initialize from first odometry
        if not self.initialized:
            self.ukf.set_state(0, x)
            self.ukf.set_state(1, y)
            self.ukf.set_state(2, theta)
            self.last_x = x
            self.last_y = y
            self.last_theta = theta
            self.last_odom_time = current_time
            self.initialized = True
            self.get_logger().info(f'UKF initialized at ({x:.2f}, {y:.2f}, {theta:.2f})')
            self.publish_estimate(current_time)
            return
        
        # Calculate motion
        dt = (current_time - self.last_odom_time).nanoseconds / 1e9
        dx = x - self.last_x
        dy = y - self.last_y
        dtheta = self.normalize_angle(theta - self.last_theta)
        
        # UKF Predict
        self.ukf.predict(dt, dx, dy, dtheta)
        
        # Update for next callback
        self.last_x = x
        self.last_y = y
        self.last_theta = theta
        self.last_odom_time = current_time
        
        # Publish estimate
        self.publish_estimate(current_time)
    
    def landmarks_callback(self, msg):
        """Handle landmark observations - UKF update step."""
        if not self.initialized:
            return
        
        # Parse observations
        observations = []
        for point in pc2.read_points(msg, field_names=('x', 'y', 'id'), skip_nans=True):
            landmark_id = int(point[2])
            obs_x = float(point[0])
            obs_y = float(point[1])
            observations.append((landmark_id, obs_x, obs_y))
        
        # UKF Update
        if observations:
            self.ukf.update(observations)
    
    def publish_estimate(self, timestamp):
        """Publish estimated odometry."""
        state = self.ukf.get_state()
        
        msg = Odometry()
        msg.header.stamp = timestamp.to_msg()
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'robot_estimated'
        
        # Position
        msg.pose.pose.position.x = state[0]
        msg.pose.pose.position.y = state[1]
        msg.pose.pose.position.z = 0.0
        
        # Orientation
        q = tf_transformations.quaternion_from_euler(0, 0, state[2])
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]
        
        # Velocity
        msg.twist.twist.linear.x = state[3]
        msg.twist.twist.linear.y = state[4]
        
        self.est_odom_pub.publish(msg)
    
    @staticmethod
    def quaternion_to_yaw(q):
        """Convert quaternion to yaw angle."""
        return tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
    
    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-π, π]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


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

