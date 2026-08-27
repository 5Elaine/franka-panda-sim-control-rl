import time

import mujoco
import mujoco.viewer


MODEL_PATH = "falling_ball.xml"
SIMULATION_DURATION = 4.0
START_DELAY = 2.0


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

    print("Viewer opened.")
    print(f"The ball will start falling after {START_DELAY:.1f} seconds.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 先保留初始悬空状态，便于观察和录屏。
        delay_start = time.time()

        while viewer.is_running() and time.time() - delay_start < START_DELAY:
            viewer.sync()
            time.sleep(0.01)

        print("Physics simulation started.")

        wall_start = time.time()
        next_print_time = 0.0

        while viewer.is_running() and data.time < SIMULATION_DURATION:
            step_start = time.time()

            mujoco.mj_step(model, data)

            if data.time >= next_print_time:
                print(
                    f"t={data.time:5.2f} s | "
                    f"z={data.xpos[ball_id][2]:.4f} m | "
                    f"contacts={data.ncon}"
                )
                next_print_time += 0.5

            viewer.sync()

            elapsed = time.time() - step_start
            remaining = model.opt.timestep - elapsed

            if remaining > 0:
                time.sleep(remaining)

        wall_elapsed = time.time() - wall_start

    print("Viewer closed.")
    print(f"simulated time: {data.time:.3f} s")
    print(f"wall-clock time: {wall_elapsed:.3f} s")
    print(f"final ball z: {data.xpos[ball_id][2]:.4f} m")


if __name__ == "__main__":
    main()
