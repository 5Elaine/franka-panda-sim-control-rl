#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MUJOCO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"

BRIDGE_SCRIPT="$SCRIPT_DIR/mujoco_panda_bridge.py"

URDF_FILE="${PANDA_URDF:-/opt/ros/humble/share/moveit_resources_panda_description/urdf/panda.urdf}"

RVIZ_CONFIG="$MUJOCO_DIR/config/mujoco_panda.rviz"

BRIDGE_LOG="${TMPDIR:-/tmp}/mujoco_bridge.log"
RSP_LOG="${TMPDIR:-/tmp}/robot_state_publisher.log"


cleanup() {
    echo
    echo "Stopping MuJoCo–ROS 2 visualization system..."

    if [[ -n "${BRIDGE_PID:-}" ]]; then
        kill "$BRIDGE_PID" 2>/dev/null || true
    fi

    if [[ -n "${RSP_PID:-}" ]]; then
        kill "$RSP_PID" 2>/dev/null || true
    fi

    wait "${BRIDGE_PID:-}" 2>/dev/null || true
    wait "${RSP_PID:-}" 2>/dev/null || true

    echo "System stopped."
}


trap cleanup EXIT INT TERM


source /opt/ros/humble/setup.bash

# Enable undefined-variable checking only after ROS setup.
set -u


for required_file in \
    "$PYTHON_EXECUTABLE" \
    "$BRIDGE_SCRIPT" \
    "$URDF_FILE" \
    "$RVIZ_CONFIG"
do
    if [[ ! -e "$required_file" ]]; then
        echo "[ERROR] Required file not found:"
        echo "        $required_file"
        exit 1
    fi
done


echo "===== MuJoCo–ROS 2 RViz system ====="
echo "Bridge:      $BRIDGE_SCRIPT"
echo "URDF:        $URDF_FILE"
echo "RViz config: $RVIZ_CONFIG"
echo


echo "[1/3] Starting MuJoCo Panda Bridge..."

"$PYTHON_EXECUTABLE" "$BRIDGE_SCRIPT" \
    >"$BRIDGE_LOG" 2>&1 &

BRIDGE_PID=$!

sleep 1

if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
    echo "[ERROR] Bridge failed to start."
    cat "$BRIDGE_LOG"
    exit 1
fi


echo "[2/3] Starting robot_state_publisher..."

ros2 run robot_state_publisher \
    robot_state_publisher \
    "$URDF_FILE" \
    >"$RSP_LOG" 2>&1 &

RSP_PID=$!

sleep 1

if ! kill -0 "$RSP_PID" 2>/dev/null; then
    echo "[ERROR] robot_state_publisher failed to start."
    cat "$RSP_LOG"
    exit 1
fi


echo "[3/3] Starting RViz..."
echo
echo "[PASS] Background nodes started."
echo "Close RViz or press Ctrl+C here to stop the system."

rviz2 -d "$RVIZ_CONFIG"
