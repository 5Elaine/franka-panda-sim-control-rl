import os
from pathlib import Path
import numpy as np
import mujoco
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MENAGERIE_ROOT = Path(os.environ.get(
    "MUJOCO_MENAGERIE_PATH",
    PROJECT_ROOT / "third_party" / "mujoco_menagerie",
)).expanduser()
MODEL_PATH = str(MENAGERIE_ROOT / "franka_emika_panda" / "scene.xml")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classical" / "static_hold"

os.makedirs(OUTPUT_DIR, exist_ok=True)

JOINT_ID = 1
ACTUATOR_ID = 1

TARGET = -0.5

KP = 200.0
KD = 20.0

TORQUE_LIMIT = 87.0

SIM_TIME = 3.0

results = {}


def gravity_torque(model, helper, qpos):

    helper.qpos[:] = qpos
    helper.qvel[:] = 0.0
    helper.qacc[:] = 0.0

    mujoco.mj_forward(model, helper)

    return float(
        helper.qfrc_bias[JOINT_ID]
    )


def run_case(use_gravity):

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)

    data = mujoco.MjData(model)
    helper = mujoco.MjData(model)

    # joint2 -> direct torque actuator
    model.actuator_gainprm[ACTUATOR_ID, :] = 0.0
    model.actuator_gainprm[ACTUATOR_ID, 0] = 1.0

    model.actuator_biasprm[ACTUATOR_ID, :] = 0.0

    model.actuator_ctrlrange[ACTUATOR_ID] = [
        -TORQUE_LIMIT,
        TORQUE_LIMIT
    ]

    model.actuator_ctrllimited[ACTUATOR_ID] = 1

    model.actuator_forcerange[ACTUATOR_ID] = [
        -TORQUE_LIMIT,
        TORQUE_LIMIT
    ]

    model.actuator_forcelimited[ACTUATOR_ID] = 1

    # --------------------------------------------------------
    # Start EXACTLY at target
    # --------------------------------------------------------

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    data.qpos[JOINT_ID] = TARGET

    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    steps = int(SIM_TIME / dt)

    time_list = []
    q_list = []
    error_list = []
    pd_list = []
    gravity_list = []
    total_list = []

    for _ in range(steps):

        q = float(data.qpos[JOINT_ID])
        dq = float(data.qvel[JOINT_ID])

        error = TARGET - q

        tau_pd = (
            KP * error
            - KD * dq
        )

        tau_g = gravity_torque(
            model,
            helper,
            data.qpos
        )

        if use_gravity:
            tau_cmd = tau_pd + tau_g
        else:
            tau_cmd = tau_pd

        tau_cmd = float(
            np.clip(
                tau_cmd,
                -TORQUE_LIMIT,
                TORQUE_LIMIT
            )
        )

        data.ctrl[ACTUATOR_ID] = tau_cmd

        # other joints use original position servos
        data.ctrl[0] = 0.0
        data.ctrl[2:7] = 0.0

        mujoco.mj_step(model, data)

        time_list.append(data.time)
        q_list.append(data.qpos[JOINT_ID])
        error_list.append(
            TARGET - data.qpos[JOINT_ID]
        )

        pd_list.append(tau_pd)

        gravity_list.append(
            tau_g if use_gravity else 0.0
        )

        total_list.append(tau_cmd)

    return {
        "time": np.asarray(time_list),
        "q": np.asarray(q_list),
        "error": np.asarray(error_list),
        "pd": np.asarray(pd_list),
        "gravity": np.asarray(gravity_list),
        "total": np.asarray(total_list),
    }


results["PD_only"] = run_case(False)
results["PD_plus_gravity"] = run_case(True)


print("\n========== STATIC GRAVITY HOLD ==========")

print(f"Initial joint2 position : {TARGET:.6f} rad")
print(f"Kp                      : {KP}")
print(f"Kd                      : {KD}")
print()

for name, r in results.items():

    final_q = r["q"][-1]
    final_error = abs(
        TARGET - final_q
    )

    max_drift = np.max(
        np.abs(
            r["q"] - TARGET
        )
    )

    final_torque = r["total"][-1]

    print(f"--- {name} ---")
    print(
        f"Final q        : "
        f"{final_q:.6f} rad"
    )
    print(
        f"Final error    : "
        f"{final_error:.6f} rad"
    )
    print(
        f"Maximum drift  : "
        f"{max_drift:.6f} rad"
    )
    print(
        f"Final torque   : "
        f"{final_torque:.6f} N·m"
    )
    print()


# ------------------------------------------------------------
# Position figure
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

for name, r in results.items():

    plt.plot(
        r["time"],
        r["q"],
        label=name
    )

plt.axhline(
    TARGET,
    linestyle="--",
    label="Target"
)

plt.xlabel("Time (s)")
plt.ylabel("Joint 2 Position (rad)")
plt.title(
    "Static Holding: PD vs Gravity Compensation"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

fig_path = os.path.join(
    OUTPUT_DIR,
    "static_hold_position.png"
)

plt.savefig(
    fig_path,
    dpi=200
)

print("[PASS] static holding experiment finished")
print(f"\nFigure saved to:\n{fig_path}")
