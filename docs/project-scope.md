# Scope, limitations, and reproducibility notes

## Included

- ROS 2 packages for Panda motion examples, Gazebo bringup, planning-scene operations, and automated pick-and-place.
- MuJoCo examples, ROS 2 bridge, cross-simulator benchmark scripts, controller implementations, experiment CSVs, figures, and PPO v4/v5 checkpoints.
- Public-facing screenshots and the full experiment summaries used in the README.

## Explicit limitations

- The MPC and PPO experiments control **joint 2 only**.
- No real Franka robot was used; results are simulation-only.
- No vision, learned grasp perception, VLA policy, sim-to-real transfer, or hardware safety validation is claimed.
- Gazebo/MuJoCo comparisons include stack-level differences and should not be read as isolated physics-engine benchmarks.
- The original PPO v3 environment and checkpoint were not present in the curated source bundle. Its numbers are retained as historical comparison results, while the included v4/v5 checkpoints are directly inspectable.

## Reproducibility boundary

The committed CSVs and figures reproduce the recorded project evidence. Re-running exact trajectories may vary with ROS package versions, solver versions, CPU scheduling, and simulator timing. The external MuJoCo Menagerie model and Franka description packages are not vendored.
