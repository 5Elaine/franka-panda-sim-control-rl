# Architecture

## ROS 2 / Gazebo planning stack

```text
one-command launch
  ├─ Gazebo + ros2_control
  ├─ robot_state_publisher
  ├─ MoveIt move_group
  ├─ RViz
  └─ pick_place_state_machine
       ├─ PreflightChecker (11 checks)
       ├─ SceneManager (world/attached object synchronization)
       ├─ MotionManager (OMPL + Cartesian execution)
       └─ GripperManager (action control + measured closure validation)
```

The automated task uses 15 states: initialize scene, open gripper, move to pre-grasp, Cartesian approach, close, attach, lift, transport, lower, detach, release, synchronize the placed scene, retreat, return home, and complete.

The important design choice is to treat Gazebo physics and the MoveIt Planning Scene as two states that must be synchronized explicitly. The cube is moved from a world collision object to an attached object during grasp, then returned to the world and updated after physical release.

## MuJoCo / ROS 2 bridge

```text
/joint_command ──> MuJoCo bridge ──> Panda actuator controls
                         │
                         ├──> /joint_states ──> robot_state_publisher ──> TF
                         └──> RViz RobotModel
```

The bridge runs MuJoCo as the dynamics source while exposing the robot through standard ROS 2 messages. A separate benchmark path records the same target/home command sequence in MuJoCo and Gazebo.

## Control progression

```text
PD
 └─ gravity compensation
     └─ inverse dynamics
         └─ computed torque
             └─ constrained error-state MPC

PPO v4 (safe action shaping)
 └─ PPO v5 (domain randomization)
     └─ out-of-distribution evaluation
```

Classical, MPC, and RL experiments all focus on Panda joint 2 so that controller behavior and constraints can be compared clearly under a controlled setup.
