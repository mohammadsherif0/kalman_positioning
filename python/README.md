# UKF Robot Localization - Python Implementation

This is a complete Python implementation of an Unscented Kalman Filter (UKF) for robot localization using landmark observations and odometry.

## Project Structure

```
python/
├── kalman_positioning/          # Main package
│   ├── __init__.py
│   ├── ukf.py                   # UKF implementation (Tasks A1-A6)
│   ├── landmark_manager.py      # Landmark management
│   └── simulator.py             # Robot simulator
├── experiments/                 # Experimental tasks
│   ├── __init__.py
│   ├── task_b1_process_noise.py        # Task B1
│   ├── task_b2_measurement_noise.py    # Task B2
│   ├── task_b3_simulation_params.py    # Task B3
│   └── results/                        # Generated results
├── run_simulation.py            # Basic simulation script
├── run_all_experiments.py       # Master experiment script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Features

### Part A: UKF Implementation (50 Points)

All tasks A1-A6 are **fully implemented** in `kalman_positioning/ukf.py`:

✅ **Task A1: Constructor and Initialization (10 pts)**
- State initialization: `x = [0, 0, 0, 0, 0]`
- Process noise matrix Q
- Measurement noise matrix R
- Covariance matrix P
- UKF parameters: λ, γ, weights

✅ **Task A2: Sigma Point Generation (8 pts)**
- Cholesky decomposition
- 2n+1 sigma points generation
- Symmetric point distribution

✅ **Task A3: Process Model (7 pts)**
- Position update: x' = x + Δx, y' = y + Δy
- Orientation update with normalization
- Velocity calculation

✅ **Task A4: Measurement Model (8 pts)**
- Relative position calculation
- World-to-robot frame transformation
- Landmark observation prediction

✅ **Task A5: Predict Step (10 pts)**
- Sigma point propagation
- Weighted mean calculation
- Covariance prediction
- Process noise addition

✅ **Task A6: Update Step (7 pts)**
- Measurement transformation
- Innovation covariance
- Kalman gain calculation
- State and covariance update

### Part B: Experimental Analysis (30 Points)

All experimental tasks are implemented:

✅ **Task B1: Process Noise Variation (10 pts)**
- 5 simulations with varying process noise
- RMSE, convergence, and stability analysis
- Comprehensive plots and optimal parameter identification

✅ **Task B2: Measurement Noise Variation (10 pts)**
- 5 simulations with varying measurement noise
- Convergence speed and tracking accuracy analysis
- Comparison with process noise effects

✅ **Task B3: Simulation Parameter Variation (10 pts)**
- 5 scenarios: Baseline, Large curve, High noise, Small/Large observation radius
- Impact analysis of trajectory, sensor noise, and observation radius
- Comprehensive table and visualization

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Navigate to the python directory:
```bash
cd /Users/mohammadsherif/kalman_positioning/python
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy scipy matplotlib pandas
```

## Usage

### Quick Start: Basic Simulation

Run a single simulation with default parameters:

```bash
python run_simulation.py
```

This will:
- Simulate a robot moving in a circular trajectory
- Apply UKF for localization using noisy odometry and landmark observations
- Display visualization showing true vs estimated trajectory
- Print performance metrics (RMSE, convergence time, etc.)

### Running Individual Experiments

**Task B1: Process Noise Variation**
```bash
python experiments/task_b1_process_noise.py
```

**Task B2: Measurement Noise Variation**
```bash
python experiments/task_b2_measurement_noise.py
```

**Task B3: Simulation Parameter Variation**
```bash
python experiments/task_b3_simulation_params.py
```

### Running All Experiments

Run all three experimental tasks in sequence:

```bash
python run_all_experiments.py
```

This will:
1. Run Task B1 and find optimal process noise
2. Run Task B2 using optimal process noise from B1
3. Run Task B3 using optimal parameters from B1 and B2
4. Generate comprehensive reports and visualizations
5. Save all results to `experiments/results/`

**Estimated time: 5-10 minutes**

## Customization

### Adjusting Simulation Parameters

Edit `run_simulation.py` and modify the function call at the bottom:

```python
results = run_simulation(
    duration=30.0,              # Simulation duration (seconds)
    dt=0.1,                     # Time step (seconds)
    process_noise_xy=1e-4,      # UKF process noise for position
    process_noise_theta=1e-4,   # UKF process noise for orientation
    measurement_noise_xy=0.01,  # UKF measurement noise
    odometry_noise_xy=0.1,      # Simulator odometry noise
    observation_noise=0.1,      # Simulator landmark noise
    observation_radius=5.0,     # Landmark observation radius (m)
    visualize=True              # Show plots
)
```

### Creating Custom Scenarios

You can create custom scenarios by modifying the simulator:

```python
from kalman_positioning import UKF, LandmarkManager
from kalman_positioning.simulator import RobotSimulator, generate_grid_landmarks

# Create custom landmarks
landmarks = {
    0: (5.0, 5.0),
    1: (10.0, 5.0),
    2: (10.0, 10.0),
    3: (5.0, 10.0)
}

# Initialize simulator and UKF
simulator = RobotSimulator(landmarks, ...)
ukf = UKF(process_noise_xy=1e-4, ...)
ukf.set_landmarks(landmarks)

# Your custom simulation loop
```

## Understanding the Results

### Output Files

All experiment results are saved to `experiments/results/`:

- **`task_b1_process_noise_variation.png`**: Plots showing RMSE, convergence, and stability vs process noise
- **`task_b2_measurement_noise_variation.png`**: Plots showing tracking accuracy vs measurement noise
- **`task_b3_simulation_parameter_variation.png`**: Comprehensive comparison of 5 scenarios
- **`task_b3_results_table.csv`**: Detailed numerical results for all scenarios

### Key Metrics

- **RMSE**: Root Mean Square Error - overall position accuracy
- **Convergence Time**: Time to reach <10cm error threshold
- **Stability**: Standard deviation of error after convergence
- **Tracking Accuracy**: Mean error in steady-state (last 25% of trajectory)

### Interpretation

**Low RMSE + Fast Convergence = Good Performance**

**Optimal Parameters** (typical values):
- Process Noise (XY): ~1e-4 to 1e-3
- Process Noise (Theta): ~1e-4 to 1e-3
- Measurement Noise: ~0.01 to 0.05

## Algorithm Details

### UKF State Vector

```
x = [x, y, theta, vx, vy]^T
```

Where:
- `x, y`: Position in world frame (m)
- `theta`: Orientation (rad, -π to π)
- `vx, vy`: Velocities in world frame (m/s)

### UKF Parameters

- **Alpha (α)**: 1e-3 - Spread of sigma points
- **Beta (β)**: 2.0 - For Gaussian distributions
- **Kappa (κ)**: 0.0 - Secondary scaling parameter
- **Lambda (λ)**: α² × (n + κ) - n
- **Gamma (γ)**: √(n + λ)

### Sigma Points

The UKF generates **2n+1 = 11** sigma points:
- 1 at the mean
- 5 in positive directions
- 5 in negative directions

### Process Model

```
x' = x + Δx
y' = y + Δy
θ' = normalize(θ + Δθ)
vx' = Δx / Δt
vy' = Δy / Δt
```

### Measurement Model

Transforms landmark from world frame to robot frame:

```
[rel_x]   [ cos(θ)  sin(θ)] [lx - x]
[rel_y] = [-sin(θ)  cos(θ)] [ly - y]
```

## Troubleshooting

### Common Issues

**1. Import errors**
```bash
ModuleNotFoundError: No module named 'numpy'
```
Solution: Install requirements: `pip install -r requirements.txt`

**2. Plots not showing**

If matplotlib doesn't display plots, you can save them instead:
```python
results = run_simulation(..., visualize=False, save_plot='output.png')
```

**3. Slow performance**

Reduce simulation duration or increase time step:
```python
results = run_simulation(duration=10.0, dt=0.2)
```

**4. Poor convergence**

Try adjusting the noise parameters:
- Increase process noise if filter is too confident in predictions
- Decrease measurement noise if observations are accurate

## Technical Notes

### Coordinate Frames

- **World Frame**: Fixed reference frame where landmarks are defined
- **Robot Frame**: Moving frame attached to robot, x-axis forward, y-axis left

### Angle Normalization

All angles are normalized to [-π, π] to avoid discontinuities.

### Numerical Stability

- Cholesky decomposition with regularization for ill-conditioned matrices
- Symmetric covariance matrix enforcement: P = (P + P^T) / 2

## References

1. Wan, E. A., & Van Der Merwe, R. (2000). "The Unscented Kalman Filter for Nonlinear Estimation"
2. Thrun, S., Burgard, W., & Fox, D. (2005). "Probabilistic Robotics"

## License

Apache-2.0

## Author

Implementation by Mohammad Sherif for the Kalman Positioning assignment.

## Support

For issues or questions, refer to the code comments or contact the course instructor.

