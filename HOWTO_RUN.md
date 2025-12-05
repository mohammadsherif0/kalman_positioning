# How to Build and Run the C++ UKF Implementation

## ✅ Completed Tasks

- **A1**: UKF Constructor - State, Q, R, P, weights initialized with positive weights
- **A2**: Sigma Points - Generated using eigenvalue decomposition (NO Cholesky)
- **A3**: Process Model - Position, orientation, velocity update
- **A4**: Measurement Model - Landmark to robot frame transformation
- **A5**: Predict Step - Sigma point propagation and covariance update
- **A6**: Update Step - Kalman gain, innovation, state/covariance update
- **Integration**: Fully integrated into positioning_node.cpp

## 🚀 How to Run on Your VM

### Terminal 1: Build the C++ Package

```bash
cd ~/ros2_ws
colcon build --packages-select kalman_positioning
source install/setup.bash
```

### Terminal 2: Run Fake Robot

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch fake_robot fake_robot.launch.py
```

### Terminal 3: Run UKF Positioning Node

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run kalman_positioning positioning_node \
  --ros-args \
  -p landmarks_csv_path:=/home/mohammad/ros2_ws/src/kalman_positioning/landmarks.csv \
  -p process_noise_xy:=0.01 \
  -p process_noise_theta:=0.01 \
  -p measurement_noise_xy:=0.01
```

### Terminal 4: Run RViz2

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run rviz2 rviz2
```

## 📊 What You Should See in RViz2

1. **Red Path** (`/robot_noisy`): Noisy robot odometry
2. **Blue Path** (`/robot_estimated_odometry`): Smooth filtered estimate from UKF
3. **Green Markers**: Landmarks on the grid

The **blue path should track closely to the true robot position**, filtering out the noise from the red path!

## 🔧 Key Parameters to Experiment With (Task B)

Adjust these in Terminal 3 command:

- `process_noise_xy`: Process noise for x,y position (default: 0.01)
- `process_noise_theta`: Process noise for orientation (default: 0.01)
- `measurement_noise_xy`: Landmark measurement noise (default: 0.01)

**Example: Higher process noise (faster adaptation)**
```bash
ros2 run kalman_positioning positioning_node --ros-args \
  -p landmarks_csv_path:=/home/mohammad/ros2_ws/src/kalman_positioning/landmarks.csv \
  -p process_noise_xy:=0.1 \
  -p measurement_noise_xy:=0.01
```

## 🛠️ If Something Goes Wrong

### Reset Everything

```bash
cd ~/ros2_ws
rm -rf build/ install/ log/
colcon build
source install/setup.bash
```

### Check Topics Are Publishing

```bash
ros2 topic list
# Should see: /robot_noisy, /landmarks_observed, /robot_estimated_odometry

ros2 topic hz /robot_estimated_odometry
# Should see ~10 Hz
```

### View Live Position

```bash
ros2 topic echo /robot_estimated_odometry --field pose.pose.position
```

## 📝 Implementation Details

- **NO Cholesky**: Uses eigenvalue decomposition for numerical stability
- **Positive Weights**: All sigma point weights are positive (W₀ = 0.167)
- **Initialization**: UKF initializes from first odometry message
- **Simple & Straightforward**: Focused on core UKF algorithm without over-complication

