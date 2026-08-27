# Results and methodology

## ROS 2 and Gazebo

- Verified joint trajectory execution, TF, inverse kinematics, and a 5 cm Cartesian lift.
- Added a table and a 5 cm cube to both Gazebo and the MoveIt Planning Scene.
- Completed collision-aware pick-and-place with physical grasp validation and explicit attach/detach handling.
- The final one-command workflow passed three consecutive cold-start runs in the recorded test campaign.

## Gazebo vs MuJoCo benchmark

| Simulator | Phase | Arrival time | RMS tracking error | Peak joint speed |
|---|---|---:|---:|---:|
| Gazebo | target | 1.199 s | 0.09652 rad | 0.62486 rad/s |
| Gazebo | home | 1.039 s | 0.09500 rad | 0.62486 rad/s |
| MuJoCo | target | 0.779 s | 0.06998 rad | 0.75705 rad/s |
| MuJoCo | home | 0.781 s | 0.06969 rad | 0.75530 rad/s |

Measured publication rates were about 82.237 Hz in Gazebo and 50.409 Hz in MuJoCo. These are **system-level observations**, not a claim that one physics engine is intrinsically faster: controller tuning, command interpretation, bridge timing, and sampling rates differ.

## Controller benchmark

The consolidated source of truth is [`results/control/benchmark_summary.csv`](../results/control/benchmark_summary.csv).

### Classical and model-based control

- Gravity compensation reduced static final error from 0.128702 rad to approximately zero.
- For a sinusoidal joint-2 trajectory, RMS error improved from 0.031282 rad (PD + gravity) to 0.012329 rad (PD + inverse dynamics) and 0.008018 rad (computed torque).

### MPC

- Nominal error-state MPC v3 reached 0.007924 rad RMS.
- Average optimization time was 6.127 ms with no missed 20 ms control deadlines in the recorded run.
- Under a 35 N·m torque limit and 8 N·m/step rate limit, constrained MPC reached 0.111671 rad RMS versus 0.135545 rad for clipped computed torque.

### Reinforcement learning

- PPO v4 introduced safe action handling and reduced RMS error from the historical v3 baseline of 0.021994 rad to 0.013099 rad.
- PPO v5 added dynamics randomization and reduced average generalization RMS from 0.017704 rad to 0.011220 rad; worst-case RMS improved from 0.037045 rad to 0.018671 rad.
- In the recorded OOD suite, PPO v5 achieved 0.017948 rad average RMS, 0.035629 rad worst RMS, a worst degradation ratio of 4.98× nominal, and a 0% termination rate.

## Data policy

Raw and summary CSV files are committed for inspection. Trained PPO v4/v5 checkpoints are included. Plots in `assets/images/mujoco/control/` are generated from the recorded experiment campaign.
