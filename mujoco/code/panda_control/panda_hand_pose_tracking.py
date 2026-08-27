import csv
from pathlib import Path

import mujoco
import numpy as np


MODEL_PATH = "scene.xml"
CSV_PATH = Path("panda_hand_pose_tracking.csv")

ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
ARM_ACTUATOR_NAMES = [f"actuator{i}" for i in range(1, 8)]

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
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def quaternion_difference_angle(
    quaternion_a: np.ndarray,
    quaternion_b: np.ndarray,
) -> float:
    """
    Return the smallest angular difference between two quaternions.

    MuJoCo body quaternions use the order:
    [w, x, y, z]
    """
    quaternion_a = quaternion_a / np.linalg.norm(quaternion_a)
    quaternion_b = quaternion_b / np.linalg.norm(quaternion_b)

    dot_product = abs(float(np.dot(quaternion_a, quaternion_b)))
    dot_product = float(np.clip(dot_product, -1.0, 1.0))

    return 2.0 * np.arccos(dot_product)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    if model.nkey < 1:
        raise RuntimeError("Panda Home keyframe was not found.")

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    print("===== Available body names =====")
    body_names = []

    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        body_names.append(body_name)
        print(f"{body_id:2d} | {body_name}")

    print()

    hand_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "hand",
    )

    if hand_body_id < 0:
        raise RuntimeError(
            "Body 'hand' was not found. "
            f"Available bodies: {body_names}"
        )

    joint_ids = []
    actuator_ids = []
    qpos_addresses = []
    dof_addresses = []

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
        qpos_addresses.append(
            int(model.jnt_qposadr[joint_id])
        )
        dof_addresses.append(
            int(model.jnt_dofadr[joint_id])
        )

    actuator_ids = np.asarray(actuator_ids, dtype=int)
    qpos_addresses = np.asarray(qpos_addresses, dtype=int)
    dof_addresses = np.asarray(dof_addresses, dtype=int)

    home_target = data.ctrl[actuator_ids].copy()

    def arm_position() -> np.ndarray:
        return data.qpos[qpos_addresses].copy()

    def arm_velocity() -> np.ndarray:
        return data.qvel[dof_addresses].copy()

    def hand_pose() -> tuple[np.ndarray, np.ndarray]:
        position = data.xpos[hand_body_id].copy()
        quaternion = data.xquat[hand_body_id].copy()
        return position, quaternion

    home_position, home_quaternion = hand_pose()

    print("===== End-effector reference =====")
    print("Body name: hand")
    print("Body ID:", hand_body_id)
    print("Home position xyz:", home_position)
    print("Home quaternion wxyz:", home_quaternion)
    print()

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            ["time", "phase"]
            + [f"q{i}" for i in range(1, 8)]
            + [f"target{i}" for i in range(1, 8)]
            + [f"dq{i}" for i in range(1, 8)]
            + [
                "hand_x",
                "hand_y",
                "hand_z",
                "hand_qw",
                "hand_qx",
                "hand_qy",
                "hand_qz",
            ]
        )

        next_print_time = 0.0

        def one_step(
            target: np.ndarray,
            phase: str,
        ) -> None:
            nonlocal next_print_time

            data.ctrl[actuator_ids] = target
            mujoco.mj_step(model, data)

            positions = arm_position()
            velocities = arm_velocity()
            hand_position, hand_quaternion = hand_pose()

            writer.writerow(
                [f"{data.time:.6f}", phase]
                + positions.tolist()
                + target.tolist()
                + velocities.tolist()
                + hand_position.tolist()
                + hand_quaternion.tolist()
            )

            if data.time >= next_print_time:
                maximum_error = float(
                    np.max(np.abs(target - positions))
                )

                print(
                    f"t={data.time:6.3f} s | "
                    f"phase={phase:13s} | "
                    f"hand_xyz="
                    f"[{hand_position[0]: .4f}, "
                    f"{hand_position[1]: .4f}, "
                    f"{hand_position[2]: .4f}] m | "
                    f"max_joint_error="
                    f"{maximum_error:.6f} rad"
                )

                next_print_time += 0.5

        def hold(
            target: np.ndarray,
            duration: float,
            phase: str,
        ) -> None:
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
                progress = (
                    data.time - start_time
                ) / duration

                blend = smoothstep(progress)

                interpolated_target = (
                    (1.0 - blend) * start_target
                    + blend * end_target
                )

                one_step(interpolated_target, phase)

            data.ctrl[actuator_ids] = end_target

        print("===== Motion sequence =====")

        hold(home_target, 0.5, "HOME_HOLD")
        move(home_target, TARGET, 2.0, "TO_TARGET")
        hold(TARGET, 1.0, "TARGET_HOLD")

        target_position, target_quaternion = hand_pose()

        move(TARGET, home_target, 2.0, "RETURN_HOME")
        hold(home_target, 1.0, "FINAL_HOLD")

    final_position, final_quaternion = hand_pose()
    final_joint_position = arm_position()
    final_joint_velocity = arm_velocity()

    target_displacement = float(
        np.linalg.norm(target_position - home_position)
    )

    return_position_error = float(
        np.linalg.norm(final_position - home_position)
    )

    return_orientation_error = float(
        quaternion_difference_angle(
            final_quaternion,
            home_quaternion,
        )
    )

    maximum_joint_error = float(
        np.max(
            np.abs(home_target - final_joint_position)
        )
    )

    maximum_joint_speed = float(
        np.max(np.abs(final_joint_velocity))
    )

    print()
    print("===== Pose comparison =====")
    print("Home hand xyz:", home_position)
    print("Target hand xyz:", target_position)
    print("Final hand xyz:", final_position)
    print(
        "Home-to-target displacement:",
        f"{target_displacement:.6f} m",
    )
    print(
        "Return position error:",
        f"{return_position_error:.6f} m",
    )
    print(
        "Return orientation error:",
        f"{return_orientation_error:.6f} rad",
    )

    print()
    print("===== Joint-state validation =====")
    print(
        "Maximum final joint error:",
        f"{maximum_joint_error:.6f} rad",
    )
    print(
        "Maximum final joint speed:",
        f"{maximum_joint_speed:.6f} rad/s",
    )
    print("CSV saved to:", CSV_PATH.resolve())

    passed = (
        maximum_joint_error < 0.02
        and maximum_joint_speed < 0.05
        and return_position_error < 0.01
        and return_orientation_error < 0.03
    )

    if passed:
        print(
            "[PASS] Joint motion and hand-pose "
            "feedback are consistent."
        )
    else:
        print(
            "[WARN] Final state exceeded one or "
            "more tolerances."
        )


if __name__ == "__main__":
    main()
