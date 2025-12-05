#!/usr/bin/env python3
"""
CLI helper to evaluate odometry bags for Part B experiments and optionally
auto-run sweeps (B1/B2/B3) by launching the node, recording bags, and
computing metrics.

Features:
- Compute position RMSE between an estimated odom topic and a reference topic.
- Estimate convergence time (time until error stays below a threshold).
- Report post-convergence stability (std of error).
- Optional plot of error over time (requires matplotlib).

Example:
  ./experiment_cli.py \
    --bag /path/to/run1 \
    --est-topic /robot_estimated_odometry \
    --ref-topic /robot_noisy \
    --convergence-threshold 0.5 \
    --plot
"""

import argparse
import bisect
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


@dataclass
class Series:
    stamps: List[float]
    positions: List[np.ndarray]


def load_topic_series(bag_path: str, topic: str) -> Series:
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                                    output_serialization_format="cdr")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in topic_types:
        raise RuntimeError(f"Topic '{topic}' not found in bag {bag_path}")
    msg_type = get_message(topic_types[topic])

    stamps: List[float] = []
    positions: List[np.ndarray] = []

    while reader.has_next():
        topic_name, data, t = reader.read_next()
        if topic_name != topic:
            continue
        msg = deserialize_message(data, msg_type)
        stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        positions.append(np.array([msg.pose.pose.position.x, msg.pose.pose.position.y], dtype=np.float64))

    if not stamps:
        raise RuntimeError(f"No messages on topic '{topic}' in bag {bag_path}")

    return Series(stamps=stamps, positions=positions)


def align_and_errors(est: Series, ref: Series, max_dt: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    ref_times = ref.stamps
    errors = []
    times = []

    for t_est, p_est in zip(est.stamps, est.positions):
        idx = bisect.bisect_left(ref_times, t_est)
        candidates = []
        if idx < len(ref_times):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        best = None
        best_dt = None
        for j in candidates:
            dt = abs(ref_times[j] - t_est)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best = j
        if best is None or best_dt is None or best_dt > max_dt:
            continue
        err = np.linalg.norm(p_est - ref.positions[best])
        errors.append(err)
        times.append(t_est)

    if not errors:
        raise RuntimeError("No aligned samples; consider increasing max_dt")

    return np.array(times), np.array(errors)


def convergence_time(times: np.ndarray, errors: np.ndarray, threshold: float, window: int = 20) -> float:
    if len(errors) < window:
        return float("nan")
    for i in range(window, len(errors)):
        if np.all(errors[i - window:i] < threshold):
            return times[i]
    return float("nan")


def compute_metrics(bag_path: str,
                    est_topic: str,
                    ref_topic: str,
                    max_dt: float,
                    conv_threshold: float,
                    conv_window: int,
                    plot: bool,
                    quiet: bool = False) -> Dict[str, float]:
    est = load_topic_series(bag_path, est_topic)
    ref = load_topic_series(bag_path, ref_topic)
    times, errors = align_and_errors(est, ref, max_dt=max_dt)

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mean_err = float(np.mean(errors))
    std_err = float(np.std(errors))

    t_conv = convergence_time(times, errors, threshold=conv_threshold, window=conv_window)
    post_mask = errors < conv_threshold
    post_std = float(np.std(errors[post_mask])) if np.any(post_mask) else float("nan")

    metrics = {
        "bag": bag_path,
        "samples": len(errors),
        "rmse": rmse,
        "mean_error": mean_err,
        "std_error": std_err,
        "convergence_time": t_conv,
        "post_convergence_std": post_std,
    }

    if not quiet:
        print(f"Bag: {bag_path}")
        print(f"  Samples aligned: {len(errors)}")
        print(f"  RMSE: {rmse:.4f} m")
        print(f"  Mean error: {mean_err:.4f} m, Std: {std_err:.4f} m")
        print(f"  Convergence time (first sustained < {conv_threshold} m over {conv_window} samples): {t_conv:.3f} s")
        print(f"  Post-convergence std (errors < {conv_threshold} m): {post_std:.4f} m")

    if plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping plot", file=sys.stderr)
            return metrics
        plt.figure()
        plt.plot(times - times[0], errors, label="pos error (m)")
        plt.axhline(conv_threshold, color="red", linestyle="--", label="threshold")
        if not np.isnan(t_conv):
            plt.axvline(t_conv - times[0], color="green", linestyle="--", label="convergence")
        plt.xlabel("time (s)")
        plt.ylabel("position error (m)")
        plt.legend()
        plt.title(bag_path)
        plt.show()

    return metrics


def launch_and_record(params: Dict[str, str],
                      bag_dir: str,
                      topics: List[str],
                      duration: float) -> str:
    os.makedirs(bag_dir, exist_ok=True)
    bag_name = os.path.join(bag_dir, "bag")

    bag_cmd = ["ros2", "bag", "record", "-o", bag_name] + topics
    launch_cmd = [
        "ros2", "launch", "kalman_positioning", "positioning.launch.py",
        f"process_noise_xy:={params['process_noise_xy']}",
        f"process_noise_theta:={params['process_noise_theta']}",
        f"measurement_noise_xy:={params['measurement_noise_xy']}",
        f"observation_radius:={params['observation_radius']}",
        "log_level:=info",
    ]

    bag_proc = subprocess.Popen(bag_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
    time.sleep(1.0)  # allow bag to start
    launch_proc = subprocess.Popen(launch_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)

    time.sleep(duration)

    os.killpg(os.getpgid(launch_proc.pid), signal.SIGINT)
    os.killpg(os.getpgid(bag_proc.pid), signal.SIGINT)

    launch_proc.wait(timeout=10)
    bag_proc.wait(timeout=10)

    time.sleep(1.0)  # allow rosbag to flush to disk

    meta = os.path.join(bag_name, "metadata.yaml")
    if not os.path.exists(meta):
        raise RuntimeError(f"Bag metadata not found at {meta}; recording may have failed")

    return bag_name


def auto_sweeps(out_dir: str,
                duration: float,
                topics: List[str],
                max_dt: float,
                conv_threshold: float,
                conv_window: int,
                plot: bool) -> None:
    os.makedirs(out_dir, exist_ok=True)

    b1_grid = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    b2_grid = [0.0025, 0.005, 0.01, 0.02, 0.05]
    scenarios_b3 = [
        ("baseline", None, None),
        ("high_sensor_noise", None, 0.05),
        ("low_obs_radius", 3.0, None),
        ("high_obs_radius", 10.0, None),
        ("mid_obs_radius", 7.0, None),
    ]

    all_results = []

    # B1: process noise sweep
    print("\n=== B1: process noise sweep ===")
    b1_results = []
    for p in b1_grid:
        params = {
            "process_noise_xy": str(p),
            "process_noise_theta": str(p),
            "measurement_noise_xy": str(0.01),
            "observation_radius": str(5.0),
        }
        label = f"b1_proc_{p}"
        bag_path = launch_and_record(params, os.path.join(out_dir, label), topics, duration)
        metrics = compute_metrics(bag_path, "/robot_estimated_odometry", "/robot_noisy",
                                  max_dt, conv_threshold, conv_window, plot=False, quiet=True)
        metrics.update({"label": label, "stage": "B1", "process_noise_xy": p, "measurement_noise_xy": 0.01})
        b1_results.append(metrics)
        all_results.append(metrics)
        print(f"{label}: RMSE={metrics['rmse']:.4f}, t_conv={metrics['convergence_time']:.3f}")

    best_b1 = min(b1_results, key=lambda m: m["rmse"])
    best_proc = best_b1["process_noise_xy"]
    print(f"Best process_noise_xy from B1: {best_proc}")

    # B2: measurement noise sweep using best process noise
    print("\n=== B2: measurement noise sweep ===")
    b2_results = []
    for m in b2_grid:
        params = {
            "process_noise_xy": str(best_proc),
            "process_noise_theta": str(best_proc),
            "measurement_noise_xy": str(m),
            "observation_radius": str(5.0),
        }
        label = f"b2_meas_{m}"
        bag_path = launch_and_record(params, os.path.join(out_dir, label), topics, duration)
        metrics = compute_metrics(bag_path, "/robot_estimated_odometry", "/robot_noisy",
                                  max_dt, conv_threshold, conv_window, plot=False, quiet=True)
        metrics.update({"label": label, "stage": "B2", "process_noise_xy": best_proc, "measurement_noise_xy": m})
        b2_results.append(metrics)
        all_results.append(metrics)
        print(f"{label}: RMSE={metrics['rmse']:.4f}, t_conv={metrics['convergence_time']:.3f}")

    best_b2 = min(b2_results, key=lambda m: m["rmse"])
    best_meas = best_b2["measurement_noise_xy"]
    print(f"Best measurement_noise_xy from B2: {best_meas}")

    # B3: scenarios
    print("\n=== B3: scenario sweep ===")
    for name, obs_radius, meas_noise in scenarios_b3:
        m_val = meas_noise if meas_noise is not None else best_meas
        o_val = obs_radius if obs_radius is not None else 5.0
        params = {
            "process_noise_xy": str(best_proc),
            "process_noise_theta": str(best_proc),
            "measurement_noise_xy": str(m_val),
            "observation_radius": str(o_val),
        }
        label = f"b3_{name}"
        bag_path = launch_and_record(params, os.path.join(out_dir, label), topics, duration)
        metrics = compute_metrics(bag_path, "/robot_estimated_odometry", "/robot_noisy",
                                  max_dt, conv_threshold, conv_window, plot=False, quiet=True)
        metrics.update({
            "label": label,
            "stage": "B3",
            "process_noise_xy": best_proc,
            "measurement_noise_xy": m_val,
            "observation_radius": o_val,
        })
        all_results.append(metrics)
        print(f"{label}: RMSE={metrics['rmse']:.4f}, t_conv={metrics['convergence_time']:.3f}")

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate odometry bag metrics or auto-run sweeps (B1/B2/B3).")
    parser.add_argument("--auto-sweeps", action="store_true",
                        help="Run predefined sweeps for B1/B2/B3 (launch+record+metrics).")
    parser.add_argument("--bag", help="Path to rosbag2 (directory). Required if not using --auto-sweeps.")
    parser.add_argument("--est-topic", default="/robot_estimated_odometry", help="Estimated odometry topic.")
    parser.add_argument("--ref-topic", default="/robot_noisy", help="Reference or ground-truth odometry topic.")
    parser.add_argument("--max-dt", type=float, default=0.05, help="Max time difference (s) for alignment.")
    parser.add_argument("--convergence-threshold", type=float, default=0.5, help="Error threshold (m) for convergence.")
    parser.add_argument("--convergence-window", type=int, default=20, help="Samples needed under threshold to declare convergence.")
    parser.add_argument("--plot", action="store_true", help="Show error plot.")
    parser.add_argument("--out-dir", default="experiment_runs", help="Output directory for auto sweeps.")
    parser.add_argument("--run-duration", type=float, default=20.0, help="Duration (s) per auto sweep run.")
    parser.add_argument("--topics", nargs="+",
                        default=["/robot_noisy", "/robot_estimated_odometry", "/landmarks_observed"],
                        help="Topics to record during auto sweeps.")
    args = parser.parse_args()

    if args.auto_sweeps:
        auto_sweeps(
            out_dir=args.out_dir,
            duration=args.run_duration,
            topics=args.topics,
            max_dt=args.max_dt,
            conv_threshold=args.convergence_threshold,
            conv_window=args.convergence_window,
            plot=args.plot,
        )
    else:
        if not args.bag:
            parser.error("--bag is required when not using --auto-sweeps")
        compute_metrics(
            bag_path=args.bag,
            est_topic=args.est_topic,
            ref_topic=args.ref_topic,
            max_dt=args.max_dt,
            conv_threshold=args.convergence_threshold,
            conv_window=args.convergence_window,
            plot=args.plot,
        )


if __name__ == "__main__":
    main()

