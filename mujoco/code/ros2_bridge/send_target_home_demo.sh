#!/usr/bin/env bash

set -eo pipefail

# 加载 ROS 2 环境。这里不启用 set -u，
# 避免 ROS setup.bash 检查未定义变量时报错。
source /opt/ros/humble/setup.bash

echo "Checking MuJoCo Panda Bridge..."

if ! ros2 node list | grep -qx "/mujoco_panda_bridge"; then
    echo "[ERROR] /mujoco_panda_bridge is not running."
    exit 1
fi

echo
echo "[1/2] Sending Target..."

ros2 topic pub --once \
    /joint_command \
    std_msgs/msg/Float64MultiArray \
    "{data: [0.5, -0.4, 0.3, -1.8, 0.2, 1.8, -0.5]}"

echo "Waiting for Panda to reach Target..."
sleep 4

echo
echo "[2/2] Sending Home..."

ros2 topic pub --once \
    /joint_command \
    std_msgs/msg/Float64MultiArray \
    "{data: [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]}"

echo "Waiting for Panda to return Home..."
sleep 4

echo
echo "[PASS] Target -> Home command sequence completed."
