from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
import os


def generate_launch_description():
    """
    Launch file for Python UKF positioning node with ROS2 integration
    """
    
    # Get the directory containing landmarks.csv (parent of python folder)
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    landmarks_csv_default = os.path.join(pkg_dir, 'landmarks.csv')
    
    # Launch arguments - TUNED FOR BETTER CONVERGENCE
    landmarks_csv_arg = DeclareLaunchArgument(
        'landmarks_csv_path',
        default_value=landmarks_csv_default,
        description='Path to CSV file containing landmark positions'
    )
    
    process_noise_xy_arg = DeclareLaunchArgument(
        'process_noise_xy',
        default_value='0.01',  # Increased from 1e-4 for faster adaptation
        description='Process noise covariance for x,y position'
    )
    
    process_noise_theta_arg = DeclareLaunchArgument(
        'process_noise_theta',
        default_value='0.01',  # Increased from 1e-4 for faster adaptation
        description='Process noise covariance for orientation'
    )
    
    measurement_noise_xy_arg = DeclareLaunchArgument(
        'measurement_noise_xy',
        default_value='0.1',  # Increased to trust observations more initially
        description='Measurement noise covariance for landmark observations'
    )
    
    observation_radius_arg = DeclareLaunchArgument(
        'observation_radius',
        default_value='5.0',
        description='Maximum radius for observing landmarks'
    )
    
    # Python positioning node
    positioning_node = Node(
        package='kalman_positioning_python',
        executable='positioning_node',
        name='python_positioning_node',
        output='screen',
        parameters=[
            {
                'landmarks_csv_path': LaunchConfiguration('landmarks_csv_path'),
                'process_noise_xy': LaunchConfiguration('process_noise_xy'),
                'process_noise_theta': LaunchConfiguration('process_noise_theta'),
                'measurement_noise_xy': LaunchConfiguration('measurement_noise_xy'),
                'observation_radius': LaunchConfiguration('observation_radius'),
            }
        ]
    )
    
    return LaunchDescription([
        landmarks_csv_arg,
        process_noise_xy_arg,
        process_noise_theta_arg,
        measurement_noise_xy_arg,
        observation_radius_arg,
        positioning_node,
    ])
