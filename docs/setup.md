# Setup and reproduction

## ROS 2 / Gazebo

Tested in the original project environment with Ubuntu 22.04, ROS 2 Humble, Gazebo 11, MoveIt 2, and Franka ROS 2 packages.

```bash
mkdir -p ~/franka_ws/src
cd ~/franka_ws/src
git clone https://github.com/5Elaine/franka-panda-sim-control-rl.git project
cp -r project/ros2_ws/src/* .

cd ~/franka_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Run the integrated stack:

```bash
ros2 launch panda_pick_place_demo full_pick_place_demo.launch.py dry_run:=false stop_after_state:=COMPLETE
```

Run the preflight and state-sequence validation without changing the scene or robot state:

```bash
ros2 run panda_pick_place_demo pick_place_state_machine --ros-args -p dry_run:=true
```

Exact Franka package availability can vary by installation. The launch files are the authoritative record of the package names and topics used by this project.

## MuJoCo

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mujoco.txt

git clone https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
export MUJOCO_MENAGERIE_PATH="$PWD/third_party/mujoco_menagerie"
```

Example experiments:

```bash
python mujoco/control/classical/compare_computed_torque.py
python mujoco/control/mpc/joint2_mpc_v3.py
python mujoco/control/rl/test_ppo_v5_ood.py
```

Generated files are written under `outputs/` and ignored by Git.

## MuJoCo / ROS 2 / RViz bridge

Activate the Python environment containing MuJoCo, source ROS 2, then run:

```bash
bash mujoco/code/ros2_bridge/run_mujoco_rviz_bridge.sh
```

The bridge script accepts `PYTHON_EXECUTABLE` and `PANDA_URDF` environment overrides when local paths differ.
