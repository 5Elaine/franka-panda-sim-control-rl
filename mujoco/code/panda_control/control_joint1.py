import mujoco


MODEL_PATH = "scene.xml"
CONTROL_DURATION = 3.0
PRINT_INTERVAL = 0.25
TARGET_JOINT1 = 0.6


def get_name(model, object_type, object_id):
    """Safely return a MuJoCo object name."""
    name = mujoco.mj_id2name(model, object_type, object_id)
    return name if name is not None else "<unnamed>"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    if model.nkey < 1:
        raise RuntimeError("The Panda model has no keyframe.")

    # 加载官方提供的 Home keyframe。
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "joint1",
    )

    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "actuator1",
    )

    if joint_id < 0:
        raise RuntimeError("joint1 was not found.")

    if actuator_id < 0:
        raise RuntimeError("actuator1 was not found.")

    qpos_address = int(model.jnt_qposadr[joint_id])
    dof_address = int(model.jnt_dofadr[joint_id])

    initial_position = float(data.qpos[qpos_address])
    initial_velocity = float(data.qvel[dof_address])

    print("===== Joint and actuator mapping =====")
    print("Joint ID:", joint_id)
    print("Joint name:", get_name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_id,
    ))
    print("qpos address:", qpos_address)
    print("qvel address:", dof_address)
    print("Actuator ID:", actuator_id)
    print("Actuator name:", get_name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_id,
    ))
    print("Actuator ctrlrange:", model.actuator_ctrlrange[actuator_id])
    print()

    print("===== Initial state =====")
    print(f"joint1 position: {initial_position:.6f} rad")
    print(f"joint1 velocity: {initial_velocity:.6f} rad/s")
    print("Initial ctrl:", data.ctrl.copy())
    print()

    # 保留其他七个 actuator 的 Home 控制值，
    # 只修改 actuator1 的目标。
    data.ctrl[actuator_id] = TARGET_JOINT1

    print("===== Control started =====")
    print(f"joint1 target: {TARGET_JOINT1:.6f} rad")

    next_print_time = 0.0

    while data.time < CONTROL_DURATION:
        mujoco.mj_step(model, data)

        if data.time >= next_print_time:
            position = float(data.qpos[qpos_address])
            velocity = float(data.qvel[dof_address])
            error = TARGET_JOINT1 - position

            print(
                f"t={data.time:5.3f} s | "
                f"q={position:8.5f} rad | "
                f"dq={velocity:9.5f} rad/s | "
                f"error={error:9.6f} rad"
            )

            next_print_time += PRINT_INTERVAL

    final_position = float(data.qpos[qpos_address])
    final_velocity = float(data.qvel[dof_address])
    final_error = TARGET_JOINT1 - final_position

    print()
    print("===== Final result =====")
    print(f"target: {TARGET_JOINT1:.6f} rad")
    print(f"final position: {final_position:.6f} rad")
    print(f"final velocity: {final_velocity:.6f} rad/s")
    print(f"final error: {final_error:.6f} rad")

    if abs(final_error) < 0.02:
        print("[PASS] joint1 reached the target.")
    else:
        print("[WARN] joint1 did not fully reach the target.")


if __name__ == "__main__":
    main()
