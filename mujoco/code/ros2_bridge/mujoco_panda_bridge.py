from pathlib import Path
from typing import List

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


MODEL_PATH = (
    Path.home()
    / "mujoco_learning"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)

# MuJoCo 模型内部使用的关节名称。
MUJOCO_ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
MUJOCO_FINGER_JOINT_NAMES = [
    "finger_joint1",
    "finger_joint2",
]
MUJOCO_JOINT_NAMES = (
    MUJOCO_ARM_JOINT_NAMES
    + MUJOCO_FINGER_JOINT_NAMES
)

# 对外发布的 ROS 关节名称，需要与 Panda URDF 完全一致。
ROS_ARM_JOINT_NAMES = [
    f"panda_joint{i}" for i in range(1, 8)
]
ROS_FINGER_JOINT_NAMES = [
    "panda_finger_joint1",
    "panda_finger_joint2",
]
ROS_JOINT_NAMES = (
    ROS_ARM_JOINT_NAMES
    + ROS_FINGER_JOINT_NAMES
)

ARM_ACTUATOR_NAMES = [f"actuator{i}" for i in range(1, 8)]

# ROS 定时器每 0.01 秒运行一次，也就是约 100 Hz。
ROS_TIMER_PERIOD = 0.01

# Panda 模型的 MuJoCo timestep 是 0.002 秒。
# 每次 ROS 回调内部执行 5 次物理步进：
# 5 × 0.002 = 0.01 秒。
PHYSICS_STEPS_PER_CALLBACK = 5

# 每两个 ROS 回调发布一次 JointState，即约 50 Hz。
PUBLISH_EVERY_N_CALLBACKS = 2

# actuator 目标每秒允许变化的最大弧度，
# 用于防止收到 ROS 指令后目标瞬间突变。
MAX_TARGET_RATE = 0.75


class MujocoPandaBridge(Node):
    def __init__(self) -> None:
        super().__init__("mujoco_panda_bridge")

        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"MuJoCo Panda model not found: {MODEL_PATH}"
            )

        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)

        if self.model.nkey < 1:
            raise RuntimeError("Panda Home keyframe was not found.")

        # 加载官方 Home 状态，包括 qpos、qvel 和 ctrl。
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        self.arm_actuator_ids = self._find_ids(
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            ARM_ACTUATOR_NAMES,
        )

        # 状态仍然从 MuJoCo 的内部名称中读取。
        self.published_joint_ids = self._find_ids(
            mujoco.mjtObj.mjOBJ_JOINT,
            MUJOCO_JOINT_NAMES,
        )

        self.qpos_addresses = np.asarray(
            [
                int(self.model.jnt_qposadr[joint_id])
                for joint_id in self.published_joint_ids
            ],
            dtype=int,
        )

        self.dof_addresses = np.asarray(
            [
                int(self.model.jnt_dofadr[joint_id])
                for joint_id in self.published_joint_ids
            ],
            dtype=int,
        )

        self.control_ranges = self.model.actuator_ctrlrange[
            self.arm_actuator_ids
        ].copy()

        # desired_target：ROS 用户最终要求到达的位置。
        self.desired_target = self.data.ctrl[
            self.arm_actuator_ids
        ].copy()

        # applied_target：当前实际写入 MuJoCo actuator 的目标。
        # 它会逐步接近 desired_target。
        self.applied_target = self.desired_target.copy()

        self.joint_state_publisher = self.create_publisher(
            JointState,
            "/joint_states",
            10,
        )

        self.command_subscription = self.create_subscription(
            Float64MultiArray,
            "/joint_command",
            self.command_callback,
            10,
        )

        self.timer = self.create_timer(
            ROS_TIMER_PERIOD,
            self.simulation_callback,
        )

        self.callback_count = 0

        expected_advance = (
            PHYSICS_STEPS_PER_CALLBACK
            * self.model.opt.timestep
        )

        self.get_logger().info(
            "MuJoCo Panda bridge started."
        )
        self.get_logger().info(
            f"Model: {MODEL_PATH}"
        )
        self.get_logger().info(
            f"MuJoCo timestep: {self.model.opt.timestep:.4f} s"
        )
        self.get_logger().info(
            f"Physics steps per callback: "
            f"{PHYSICS_STEPS_PER_CALLBACK}"
        )
        self.get_logger().info(
            f"Simulation advance per callback: "
            f"{expected_advance:.4f} s"
        )
        self.get_logger().info(
            f"Initial arm target: {self.desired_target}"
        )
        self.get_logger().info(
            "Subscribe: /joint_command "
            "(std_msgs/Float64MultiArray, 7 values)"
        )
        self.get_logger().info(
            "Publish: /joint_states "
            "(sensor_msgs/JointState, 9 joints)"
        )

    def _find_ids(
        self,
        object_type,
        names: List[str],
    ) -> np.ndarray:
        object_ids = []

        for name in names:
            object_id = mujoco.mj_name2id(
                self.model,
                object_type,
                name,
            )

            if object_id < 0:
                raise RuntimeError(
                    f"MuJoCo object was not found: {name}"
                )

            object_ids.append(object_id)

        return np.asarray(object_ids, dtype=int)

    def command_callback(
        self,
        message: Float64MultiArray,
    ) -> None:
        command = np.asarray(message.data, dtype=float)

        if command.shape != (7,):
            self.get_logger().error(
                "Rejected /joint_command: expected exactly "
                f"7 values, received {command.size}."
            )
            return

        if not np.all(np.isfinite(command)):
            self.get_logger().error(
                "Rejected /joint_command: all values must be finite."
            )
            return

        lower_limits = self.control_ranges[:, 0]
        upper_limits = self.control_ranges[:, 1]

        invalid = np.logical_or(
            command < lower_limits,
            command > upper_limits,
        )

        if np.any(invalid):
            invalid_indices = np.where(invalid)[0]

            for index in invalid_indices:
                self.get_logger().error(
                    f"Rejected joint{index + 1} target "
                    f"{command[index]:.4f}: allowed range is "
                    f"[{lower_limits[index]:.4f}, "
                    f"{upper_limits[index]:.4f}]."
                )

            return

        self.desired_target = command.copy()

        self.get_logger().info(
            "Accepted new arm target: "
            + np.array2string(
                self.desired_target,
                precision=4,
            )
        )

    def simulation_callback(self) -> None:
        timestep = float(self.model.opt.timestep)
        maximum_delta = MAX_TARGET_RATE * timestep

        for _ in range(PHYSICS_STEPS_PER_CALLBACK):
            target_error = (
                self.desired_target
                - self.applied_target
            )

            target_increment = np.clip(
                target_error,
                -maximum_delta,
                maximum_delta,
            )

            self.applied_target += target_increment

            self.data.ctrl[
                self.arm_actuator_ids
            ] = self.applied_target

            mujoco.mj_step(
                self.model,
                self.data,
            )

        self.callback_count += 1

        if (
            self.callback_count
            % PUBLISH_EVERY_N_CALLBACKS
            == 0
        ):
            self.publish_joint_state()

    def publish_joint_state(self) -> None:
        message = JointState()

        # 这里先使用 ROS 节点的时钟。
        # MuJoCo 的内部仿真时间仍保存在 self.data.time 中。
        message.header.stamp = (
            self.get_clock().now().to_msg()
        )

        # 发布时换成与 ROS Panda URDF 一致的名称。
        message.name = ROS_JOINT_NAMES

        message.position = (
            self.data.qpos[
                self.qpos_addresses
            ].astype(float).tolist()
        )

        message.velocity = (
            self.data.qvel[
                self.dof_addresses
            ].astype(float).tolist()
        )

        self.joint_state_publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)

    node = None

    try:
        node = MujocoPandaBridge()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.get_logger().info(
                "Shutting down MuJoCo Panda bridge."
            )
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
