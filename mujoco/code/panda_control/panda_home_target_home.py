import argparse
import csv
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


MODEL_PATH = "scene.xml"
CSV_PATH = Path("panda_home_target_home.csv")

ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
ARM_ACTUATOR_NAMES = [f"actuator{i}" for i in range(1, 8)]

# 一个位于各关节范围内的安全测试目标。
TARGET = np.array([
    0.50,
    -0.40,
    0.30,
    -1.80,
    0.20,
    1.80,
    -0.50,
])


def smoothstep(value: float) -> float:
    """生成起止速度均为零的平滑插值比例。"""
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Show the MuJoCo passive viewer.",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    if model.nkey < 1:
        raise RuntimeError("The Panda Home keyframe was not found.")

    # 加载官方 Home 状态，包括 qpos、qvel 和 ctrl。
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    joint_ids = []
    qpos_addresses = []
    dof_addresses = []
    actuator_ids = []

    for joint_name, actuator_name in zip(
        ARM_JOINT_NAMES,
        ARM_ACTUATOR_NAMES,
    ):
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_name,
        )

        if joint_id < 0:
            raise RuntimeError(f"Missing joint: {joint_name}")

        if actuator_id < 0:
            raise RuntimeError(f"Missing actuator: {actuator_name}")

        joint_ids.append(joint_id)
        actuator_ids.append(actuator_id)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))

    joint_ids = np.asarray(joint_ids, dtype=int)
    actuator_ids = np.asarray(actuator_ids, dtype=int)
    qpos_addresses = np.asarray(qpos_addresses, dtype=int)
    dof_addresses = np.asarray(dof_addresses, dtype=int)

    home_target = data.ctrl[actuator_ids].copy()

    # 检查测试目标是否超出 actuator 控制范围。
    for index, actuator_id in enumerate(actuator_ids):
        low, high = model.actuator_ctrlrange[actuator_id]

        if not low <= TARGET[index] <= high:
            raise ValueError(
                f"{ARM_ACTUATOR_NAMES[index]} target "
                f"{TARGET[index]} is outside [{low}, {high}]"
            )

    def arm_position() -> np.ndarray:
        return data.qpos[qpos_addresses].copy()

    def arm_velocity() -> np.ndarray:
        return data.qvel[dof_addresses].copy()

    print("===== Panda seven-joint mapping =====")
    for index in range(7):
        print(
            f"{ARM_JOINT_NAMES[index]:6s} | "
            f"joint_id={joint_ids[index]:2d} | "
            f"qpos_adr={qpos_addresses[index]:2d} | "
            f"dof_adr={dof_addresses[index]:2d} | "
            f"actuator={ARM_ACTUATOR_NAMES[index]}"
        )

    print()
    print("Home target:", home_target)
    print("Test target:", TARGET)
    print("Gripper ctrl preserved:", data.ctrl[7])
    print()

    csv_file = CSV_PATH.open("w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)

    writer.writerow(
        ["time", "phase"]
        + [f"q{i}" for i in range(1, 8)]
        + [f"target{i}" for i in range(1, 8)]
        + [f"dq{i}" for i in range(1, 8)]
    )

    next_print_time = 0.0
    viewer = None

    def one_step(target: np.ndarray, phase: str) -> None:
        nonlocal next_print_time

        data.ctrl[actuator_ids] = target

        wall_start = time.time()
        mujoco.mj_step(model, data)

        positions = arm_position()
        velocities = arm_velocity()
        errors = target - positions

        writer.writerow(
            [f"{data.time:.6f}", phase]
            + positions.tolist()
            + target.tolist()
            + velocities.tolist()
        )

        if data.time >= next_print_time:
            print(
                f"t={data.time:6.3f} s | "
                f"phase={phase:13s} | "
                f"max_error={np.max(np.abs(errors)):.6f} rad | "
                f"max_speed={np.max(np.abs(velocities)):.6f} rad/s"
            )
            next_print_time += 0.5

        if viewer is not None:
            viewer.sync()

            elapsed = time.time() - wall_start
            remaining = model.opt.timestep - elapsed

            if remaining > 0:
                time.sleep(remaining)

    def hold(target: np.ndarray, duration: float, phase: str) -> None:
        start_time = data.time

        while data.time - start_time < duration:
            one_step(target, phase)

    def move(
        start_target: np.ndarray,
        end_target: np.ndarray,
        duration: float,
        phase: str,
    ) -> None:
        start_time = data.time

        while data.time - start_time < duration:
            progress = (data.time - start_time) / duration
            blend = smoothstep(progress)

            interpolated_target = (
                (1.0 - blend) * start_target
                + blend * end_target
            )

            one_step(interpolated_target, phase)

        data.ctrl[actuator_ids] = end_target

    def run_sequence() -> None:
        initial_hold = 2.0 if viewer is not None else 0.5

        print("===== Sequence started =====")

        hold(home_target, initial_hold, "HOME_HOLD")
        move(home_target, TARGET, 2.0, "TO_TARGET")
        hold(TARGET, 1.0, "TARGET_HOLD")
        move(TARGET, home_target, 2.0, "RETURN_HOME")
        hold(home_target, 1.0, "FINAL_HOLD")

    if args.viewer:

        print("Opening Viewer...")
        print("The motion will start after a 2-second Home hold.")

        with mujoco.viewer.launch_passive(model, data) as viewer_handle:
            viewer = viewer_handle
            run_sequence()

        viewer = None
    else:
        run_sequence()

    csv_file.close()

    final_position = arm_position()
    final_velocity = arm_velocity()
    final_error = home_target - final_position

    print()
    print("===== Final result =====")
    print("Final position:", final_position)
    print("Home target:", home_target)
    print("Final velocity:", final_velocity)
    print(
        "Maximum final error:",
        f"{np.max(np.abs(final_error)):.6f} rad",
    )
    print(
        "Maximum final speed:",
        f"{np.max(np.abs(final_velocity)):.6f} rad/s",
    )
    print("CSV saved to:", CSV_PATH.resolve())

    passed = (
        np.max(np.abs(final_error)) < 0.02
        and np.max(np.abs(final_velocity)) < 0.05
    )

    if passed:
        print("[PASS] Home -> Target -> Home completed.")
    else:
        print("[WARN] Final state did not satisfy the tolerance.")

    # 当前虚拟机中 launch_passive 在解释器清理时可能段错误。
    # Viewer 模式下直接结束进程，避免触发已知清理问题。
    if args.viewer:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
