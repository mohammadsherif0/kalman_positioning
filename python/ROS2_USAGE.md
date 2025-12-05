# ROS2 Integration for Python UKF

## Commands to Run with ROS2 and RViz2

### 1. Build the Python ROS2 Package

```bash
cd /path/to/kalman_positioning

# Build only the Python package
colcon build --packages-select kalman_positioning_python --symlink-install

# Source the workspace
source install/setup.bash
```

### 2. Run the System (3 Terminals)

**Terminal 1: Fake Robot Simulator**
```bash
source install/setup.bash
ros2 launch fake_robot fake_robot.launch.py
```

**Terminal 2: Python UKF Positioning Node**
```bash
source install/setup.bash

# Option A: Run directly
ros2 run kalman_positioning_python positioning_node

# Option B: Use launch file with parameters
ros2 launch kalman_positioning_python python_positioning.launch.py \
    process_noise_xy:=1e-4 \
    measurement_noise_xy:=0.01
```

**Terminal 3: RViz2 Visualization**
```bash
source install/setup.bash
ros2 run rviz2 rviz2
```

### 3. RViz2 Configuration

In RViz2, add these displays:

1. **Add → Odometry → /robot_noisy** (red, shows noisy dead-reckoning)
2. **Add → Odometry → /robot_estimated_odometry** (blue, shows UKF estimate)
3. **Add → PointCloud2 → /landmarks_observed** (white points)
4. Set **Fixed Frame** to `map`

### 4. Monitor Topics

```bash
# Check if node is running
ros2 node list

# Check published topics
ros2 topic list

# Monitor estimated odometry
ros2 topic echo /robot_estimated_odometry

# Check node info
ros2 node info /python_positioning_node
```

### 5. Check Performance

```bash
# Monitor publishing rate
ros2 topic hz /robot_estimated_odometry

# Check if UKF is converging
ros2 topic echo /robot_estimated_odometry --field pose.pose.position
```

## Launch File Parameters

You can customize the UKF behavior:

```bash
ros2 launch kalman_positioning_python python_positioning.launch.py \
    landmarks_csv_path:=/path/to/landmarks.csv \
    process_noise_xy:=1e-4 \
    process_noise_theta:=1e-4 \
    measurement_noise_xy:=0.01 \
    observation_radius:=5.0
```

## Architecture

```
fake_robot node
    ↓ publishes
/robot_noisy (Odometry)
    ↓ subscribes
python_positioning_node (Python UKF)
    ↓ publishes
/robot_estimated_odometry (Odometry)
    ↓ visualizes
RViz2
```

## Comparison: C++ vs Python Node

| Aspect | C++ Node | Python Node |
|--------|----------|-------------|
| **Package** | kalman_positioning | kalman_positioning_python |
| **Executable** | positioning_node | positioning_node |
| **Algorithm** | Student implementation | ✅ **Fully implemented UKF** |
| **Matrix ops** | Eigen | NumPy (eigenvalue decomp) |
| **Performance** | Faster | Sufficient for real-time |
| **Build time** | Requires compilation | Instant (--symlink-install) |

## Troubleshooting

### Node not found
```bash
# Rebuild and source
colcon build --packages-select kalman_positioning_python --symlink-install
source install/setup.bash
```

### No landmarks loaded
```bash
# Check if landmarks.csv exists in root
ls ../landmarks.csv

# Or specify path explicitly
ros2 run kalman_positioning_python positioning_node \
    --ros-args -p landmarks_csv_path:=/full/path/to/landmarks.csv
```

### Import errors
```bash
# Install Python dependencies
pip3 install numpy scipy matplotlib pandas sensor_msgs_py
```

### Not converging
Try adjusting parameters:
- **Increase process_noise**: If filter is too confident in motion model
- **Decrease measurement_noise**: If observations are accurate
- **Increase observation_radius**: To see more landmarks

## Expected Behavior

When running correctly, you should see in RViz2:
- **Red path** (noisy): Drifting, accumulated error
- **Blue path** (estimated): Smooth, converges to true trajectory
- **Error reduction**: From ~0.5m → <0.05m within 5 seconds

The Python UKF should achieve:
- RMSE: 0.02-0.05m
- Convergence: <5 seconds
- Real-time performance: 10-100 Hz

## Next Steps

1. ✅ Run fake_robot simulator
2. ✅ Run Python UKF node
3. ✅ Open RViz2 and add displays
4. ✅ Watch the filter converge!
5. Experiment with parameters in launch file

