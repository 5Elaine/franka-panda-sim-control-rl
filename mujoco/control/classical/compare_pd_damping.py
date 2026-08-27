import os
import csv
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classical" / "pd_comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = 0.5
KP = 40.0

KD_CASES = {
    "low_damping_Kd1": 1.0,
    "baseline_Kd8": 8.0,
    "high_damping_Kd30": 30.0,
}

TORQUE_LIMIT = 87.0
SIM_TIME = 4.0

all_results = {}


def run_case(case_name, kd):

    # --------------------------------------------------------
    # Load a fresh model for each case
    # --------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # Convert joint1 actuator from built-in position servo
    # to direct torque actuator:
    #
    # actuator force ~= ctrl
    # --------------------------------------------------------
    model.actuator_gainprm[0, :] = 0.0
    model.actuator_gainprm[0, 0] = 1.0

    model.actuator_biasprm[0, :] = 0.0

    model.actuator_ctrlrange[0] = [-TORQUE_LIMIT, TORQUE_LIMIT]
    model.actuator_ctrllimited[0] = 1

    model.actuator_forcerange[0] = [-TORQUE_LIMIT, TORQUE_LIMIT]
    model.actuator_forcelimited[0] = 1

    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    steps = int(SIM_TIME / dt)

    # Joints 2-7 remain at zero with original position servo
    data.ctrl[1:7] = 0.0

    records = []

    for _ in range(steps):

        q = float(data.qpos[0])
        dq = float(data.qvel[0])

        error = TARGET - q

        # Our PD controller
        tau = KP * error - kd * dq

        tau = float(
            np.clip(
                tau,
                -TORQUE_LIMIT,
                TORQUE_LIMIT
            )
        )

        data.ctrl[0] = tau
        data.ctrl[1:7] = 0.0

        mujoco.mj_step(model, data)

        records.append([
            data.time,
            data.qpos[0],
            data.qvel[0],
            TARGET - data.qpos[0],
            tau
        ])

    arr = np.asarray(records)

    time = arr[:, 0]
    q = arr[:, 1]
    dq = arr[:, 2]
    error = arr[:, 3]
    tau = arr[:, 4]

    final_error = abs(TARGET - q[-1])

    peak_q = np.max(q)

    overshoot = max(
        0.0,
        peak_q - TARGET
    )

    peak_speed = np.max(np.abs(dq))
    peak_torque = np.max(np.abs(tau))

    threshold = 0.01
    settling_time = np.nan

    for i in range(len(error)):
        if np.all(np.abs(error[i:]) <= threshold):
            settling_time = time[i]
            break

    # Save CSV
    csv_path = os.path.join(
        OUTPUT_DIR,
        f"{case_name}.csv"
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "time_s",
            "joint1_position_rad",
            "joint1_velocity_rad_s",
            "position_error_rad",
            "command_torque_Nm"
        ])

        writer.writerows(records)

    all_results[case_name] = {
        "kd": kd,
        "time": time,
        "q": q,
        "dq": dq,
        "error": error,
        "tau": tau,
        "final_error": final_error,
        "overshoot": overshoot,
        "peak_speed": peak_speed,
        "peak_torque": peak_torque,
        "settling_time": settling_time
    }


# ============================================================
# Run all cases
# ============================================================

for name, kd in KD_CASES.items():
    run_case(name, kd)


# ============================================================
# Print summary
# ============================================================

print("\n========== PD DAMPING COMPARISON ==========")
print(f"Kp fixed at: {KP}")
print(f"Target     : {TARGET} rad\n")

for name, r in all_results.items():

    print(f"--- {name} ---")
    print(f"Kd              : {r['kd']:.2f}")
    print(f"Final error     : {r['final_error']:.6f} rad")
    print(f"Overshoot       : {r['overshoot']:.6f} rad")
    print(f"Peak speed      : {r['peak_speed']:.6f} rad/s")
    print(f"Peak torque     : {r['peak_torque']:.6f} N·m")

    if np.isnan(r["settling_time"]):
        print("Settling time   : NOT settled")
    else:
        print(f"Settling time   : {r['settling_time']:.6f} s")

    print()


# ============================================================
# Plot 1: Joint position
# ============================================================

plt.figure(figsize=(8, 5))

for name, r in all_results.items():
    plt.plot(
        r["time"],
        r["q"],
        label=f"Kd={r['kd']}"
    )

plt.axhline(
    TARGET,
    linestyle="--",
    label="Target"
)

plt.xlabel("Time (s)")
plt.ylabel("Joint 1 Position (rad)")
plt.title("PD Damping Comparison: Joint Position")
plt.legend()
plt.grid(True)

plt.tight_layout()

position_fig = os.path.join(
    OUTPUT_DIR,
    "pd_position_comparison.png"
)

plt.savefig(position_fig, dpi=200)


# ============================================================
# Plot 2: Position error
# ============================================================

plt.figure(figsize=(8, 5))

for name, r in all_results.items():
    plt.plot(
        r["time"],
        r["error"],
        label=f"Kd={r['kd']}"
    )

plt.axhline(
    0.0,
    linestyle="--"
)

plt.xlabel("Time (s)")
plt.ylabel("Position Error (rad)")
plt.title("PD Damping Comparison: Tracking Error")
plt.legend()
plt.grid(True)

plt.tight_layout()

error_fig = os.path.join(
    OUTPUT_DIR,
    "pd_error_comparison.png"
)

plt.savefig(error_fig, dpi=200)


# ============================================================
# Plot 3: Torque command
# ============================================================

plt.figure(figsize=(8, 5))

for name, r in all_results.items():
    plt.plot(
        r["time"],
        r["tau"],
        label=f"Kd={r['kd']}"
    )

plt.axhline(
    0.0,
    linestyle="--"
)

plt.xlabel("Time (s)")
plt.ylabel("Command Torque (N·m)")
plt.title("PD Damping Comparison: Torque Command")
plt.legend()
plt.grid(True)

plt.tight_layout()

torque_fig = os.path.join(
    OUTPUT_DIR,
    "pd_torque_comparison.png"
)

plt.savefig(torque_fig, dpi=200)

print("[PASS] PD comparison finished")

print("\nFigures saved to:")
print(position_fig)
print(error_fig)
print(torque_fig)
