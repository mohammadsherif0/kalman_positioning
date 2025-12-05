# Quick Start Guide

## Installation (1 minute)

```bash
cd /Users/mohammadsherif/kalman_positioning/python
pip install -r requirements.txt
```

## Run Your First Simulation (30 seconds)

```bash
python run_simulation.py
```

You'll see:
- UKF initialization messages
- Simulation progress updates
- Performance metrics (RMSE, convergence time)
- Visualization with 4 plots showing true vs estimated trajectory

**Expected Output:**
```
UKF Initialized:
  State dimension: 5
  Lambda: -0.004975
  Gamma: 2.236...
  
Simulation complete!
RMSE: ~0.02-0.05 m
Mean Error: ~0.02-0.04 m
Convergence Time: ~2-5 seconds
```

## Run All Experiments (5-10 minutes)

```bash
python run_all_experiments.py
```

This runs Tasks B1, B2, and B3 sequentially and generates all plots.

## Test Individual Components

### Test Landmark Manager

```python
from kalman_positioning import LandmarkManager

lm = LandmarkManager()
lm.load_from_csv('data/landmarks.csv')
print(f"Loaded {lm.get_num_landmarks()} landmarks")

# Get landmarks near (10, 10) within 5m radius
nearby = lm.get_landmarks_in_radius(10.0, 10.0, 5.0)
print(f"Found {len(nearby)} landmarks nearby")
```

### Test UKF

```python
from kalman_positioning import UKF
import numpy as np

# Initialize UKF
ukf = UKF(process_noise_xy=1e-4, 
          process_noise_theta=1e-4,
          measurement_noise_xy=0.01)

# Set landmarks
landmarks = {0: (5.0, 5.0), 1: (10.0, 5.0)}
ukf.set_landmarks(landmarks)

# Prediction step
ukf.predict(dt=0.1, dx=0.1, dy=0.0, dtheta=0.0)

# Update step with observation
observations = [(0, 4.9, 0.0, 0.01)]  # (id, rel_x, rel_y, cov)
ukf.update(observations)

print("State:", ukf.get_state())
print("Position:", ukf.get_position())
```

### Test Simulator

```python
from kalman_positioning.simulator import RobotSimulator, generate_grid_landmarks

landmarks = generate_grid_landmarks(grid_size=5, spacing=2.0)
sim = RobotSimulator(landmarks, observation_radius=5.0)

sim.reset(x=0.0, y=0.0, theta=0.0)
sim.move(dx=1.0, dy=0.0, dtheta=0.1, dt=0.1)

true_state = sim.get_true_state()
observations = sim.get_landmark_observations()

print(f"True state: {true_state}")
print(f"Observed {len(observations)} landmarks")
```

## Understanding the Output

### Console Output

```
Step 0/300, t=0.0s, True: (15.00, 10.00), Est: (0.00, 0.00), Error: 15.000m, Obs: 8
```

- **Step**: Simulation step number
- **t**: Time in seconds
- **True**: True robot position
- **Est**: UKF estimated position
- **Error**: Position error in meters
- **Obs**: Number of landmarks observed

### Visualization Plots

**Plot 1 (Top-Left): Robot Trajectories**
- Green line: True trajectory
- Red dashed: Noisy odometry (dead-reckoning)
- Blue line: UKF estimate
- Gray squares: Landmarks

**Plot 2 (Top-Right): Position Error Over Time**
- Blue line: Error magnitude over time
- Red dashed: Mean error

**Plot 3 (Bottom-Left): X Position**
- Comparison of true, noisy, and estimated x-coordinate

**Plot 4 (Bottom-Right): Y Position**
- Comparison of true, noisy, and estimated y-coordinate

## Common Parameters

### UKF Tuning Parameters

```python
process_noise_xy = 1e-4      # Lower = trust motion model more
process_noise_theta = 1e-4   # Lower = trust orientation more
measurement_noise_xy = 0.01  # Lower = trust observations more
```

**Rule of thumb:**
- If estimate is too smooth/slow: Increase process noise
- If estimate is too jittery: Decrease measurement noise OR increase process noise
- If not converging: Decrease measurement noise OR increase process noise

### Simulator Parameters

```python
odometry_noise_xy = 0.1      # Real-world odometry noise
odometry_noise_theta = 0.05  # Real-world orientation noise
observation_noise = 0.1      # Landmark observation noise
observation_radius = 5.0     # How far robot can see landmarks
```

## Verify Installation

Run this quick test:

```bash
python -c "from kalman_positioning import UKF, LandmarkManager; print('✓ Installation successful!')"
```

Expected output: `✓ Installation successful!`

## Next Steps

1. ✅ Run basic simulation: `python run_simulation.py`
2. ✅ Review visualization and understand the filter behavior
3. ✅ Run Task B1: `python experiments/task_b1_process_noise.py`
4. ✅ Run Task B2: `python experiments/task_b2_measurement_noise.py`
5. ✅ Run Task B3: `python experiments/task_b3_simulation_params.py`
6. ✅ Or run all at once: `python run_all_experiments.py`

## Troubleshooting

**Problem: "ModuleNotFoundError"**
```bash
pip install numpy scipy matplotlib pandas
```

**Problem: "No display"**
```python
# In run_simulation.py, change:
visualize=False, save_plot='result.png'
```

**Problem: "Simulation too slow"**
```python
# Reduce duration or increase dt:
run_simulation(duration=10.0, dt=0.2)
```

## Tips for Best Results

1. **Start with default parameters** - They're already tuned reasonably well
2. **Watch the error plot** - Should converge within 5-10 seconds
3. **Check observation count** - Should see 5-15 landmarks per step typically
4. **RMSE < 0.1m is good** - Much lower and you might be overfitting

## Questions?

Check the full README.md for detailed documentation!

