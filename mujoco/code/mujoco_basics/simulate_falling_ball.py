import mujoco


MODEL_PATH = "falling_ball.xml"
SIMULATION_DURATION = 1.5
PRINT_INTERVAL = 0.1


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    ball_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "ball",
    )

    if ball_id < 0:
        raise RuntimeError("Body 'ball' was not found in the model.")

    mujoco.mj_forward(model, data)

    print("===== Simulation start =====")
    print(f"timestep: {model.opt.timestep:.4f} s")
    print(f"initial z: {data.xpos[ball_id][2]:.4f} m")
    print(f"initial qpos: {data.qpos.copy()}")
    print(f"initial qvel: {data.qvel.copy()}")
    print()

    next_print_time = 0.0
    maximum_contacts = 0

    while data.time < SIMULATION_DURATION:
        mujoco.mj_step(model, data)
        maximum_contacts = max(maximum_contacts, data.ncon)

        if data.time >= next_print_time:
            z_position = data.xpos[ball_id][2]
            z_velocity = data.qvel[2]

            print(
                f"t={data.time:5.3f} s | "
                f"z={z_position:7.4f} m | "
                f"vz={z_velocity:8.4f} m/s | "
                f"contacts={data.ncon}"
            )

            next_print_time += PRINT_INTERVAL

    final_z = data.xpos[ball_id][2]

    print()
    print("===== Simulation complete =====")
    print(f"final z: {final_z:.4f} m")
    print(f"maximum contacts: {maximum_contacts}")

    if maximum_contacts == 0:
        raise RuntimeError("The ball never contacted the ground.")

    if not 0.06 <= final_z <= 0.10:
        raise RuntimeError(
            f"Unexpected final ball height: {final_z:.4f} m"
        )

    print("[PASS] Gravity, dynamics and ground contact are working.")


if __name__ == "__main__":
    main()
