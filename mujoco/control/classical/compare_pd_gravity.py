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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classical" / "pd_gravity"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Experiment parameters
# ============================================================

JOINT_ID = 1          # joint2
ACTUATOR_ID = 1       # actuator2

TARGET = -0.5         # rad

KP = 40.0
KD = 8.0

TORQUE_LIMIT = 87.0

SIM_TIME = 5.0

CASES = [
    "PD_only",
    "PD_plus_gravity"
]

results = {}


# ============================================================
# Helper: estimate gravity torque only
# ============================================================

def get_gravity_torque(model, gravity_data, qpos, joint_id):

    # Copy current configuration
    gravity_data.qpos[:] = qpos

    # Zero velocity and acceleration so velocity-dependent
    # Coriolis / centrifugal terms disappear.
    gravity_data.qvel[:] = 0.0
    gravity_data.qacc[:] = 0.0

    mujoco.mj_forward(model, gravity_data)

    # At zero velocity, qfrc_bias is dominated by gravity.
    return float(gravity_data.qfrc_bias[joint_id])


# ============================================================
# Run one controller case
# ============================================================

def run_case(case_name):

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)

    data = mujoco.MjData(model)
    gravity_data = mujoco.MjData(model)

    # --------------------------------------------------------
    # Convert ONLY joint2 actuator from built-in position servo
    # to direct torque control:
    #
    # actuator force ≈ ctrl
    # --------------------------------------------------------

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

    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    steps = int(SIM_TIME / dt)

    # Other arm joints retain their original position servos.
    data.ctrl[0] = 0.0
    data.ctrl[2:7] = 0.0

    records = []

    initial_gravity = get_gravity_torque(
        model,
        gravity_data,
        data.qpos,
        JOINT_ID
    )

    for _ in range(steps):

        q = float(data.qpos[JOINT_ID])
        dq = float(data.qvel[JOINT_ID])

        error = TARGET - q

        # ----------------------------------------------------
        # Feedback term
        # ----------------------------------------------------

        tau_pd = KP * error - KD * dq

        # ----------------------------------------------------
        # Model-based gravity compensation
        # ----------------------------------------------------

        tau_g = get_gravity_torque(
            model,
            gravity_data,
            data.qpos,
            JOINT_ID
        )

        if case_name == "PD_only":
            tau_cmd = tau_pd
            tau_g_used = 0.0

        elif case_name == "PD_plus_gravity":
            tau_cmd = tau_pd + tau_g
            tau_g_used = tau_g

        else:
            raise ValueError("Unknown case")

        tau_cmd = float(
            np.clip(
                tau_cmd,
                -TORQUE_LIMIT,
                TORQUE_LIMIT
            )
        )

        data.ctrl[ACTUATOR_ID] = tau_cmd

        # Other arm joints still use original position control
        data.ctrl[0] = 0.0
        data.ctrl[2:7] = 0.0

        mujoco.mj_step(model, data)

        records.append([
            data.time,
            data.qpos[JOINT_ID],
            data.qvel[JOINT_ID],
            TARGET - data.qpos[JOINT_ID],
            tau_pd,
            tau_g_used,
            tau_cmd
        ])

    arr = np.asarray(records)

    time = arr[:, 0]
    q = arr[:, 1]
    dq = arr[:, 2]
    error = arr[:, 3]
    tau_pd = arr[:, 4]
    tau_g = arr[:, 5]
    tau_cmd = arr[:, 6]

    # ========================================================
    # Metrics
    # ========================================================

    final_q = q[-1]
    final_error = abs(TARGET - final_q)

    peak_speed = np.max(np.abs(dq))
    peak_torque = np.max(np.abs(tau_cmd))

    rms_error = np.sqrt(
        np.mean(error ** 2)
    )

    # Approximate integral of squared torque:
    # a simple control-effort metric
    control_effort = np.sum(
        tau_cmd ** 2
    ) * dt

    threshold = 0.01
    settling_time = np.nan

    for i in range(len(error)):
        if np.all(np.abs(error[i:]) <= threshold):
            settling_time = time[i]
            break

    final_gravity = get_gravity_torque(
        model,
        gravity_data,
        data.qpos,
        JOINT_ID
    )

    # ========================================================
    # Save CSV
    # ========================================================

    csv_path = os.path.join(
        OUTPUT_DIR,
        f"{case_name}.csv"
    )

    with open(csv_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "time_s",
            "joint2_position_rad",
            "joint2_velocity_rad_s",
            "position_error_rad",
            "pd_torque_Nm",
            "gravity_comp_torque_Nm",
            "command_torque_Nm"
        ])

        writer.writerows(records)

    results[case_name] = {
        "time": time,
        "q": q,
        "dq": dq,
        "error": error,
        "tau_pd": tau_pd,
        "tau_g": tau_g,
        "tau_cmd": tau_cmd,

        "initial_gravity": initial_gravity,
        "final_gravity": final_gravity,

        "final_q": final_q,
        "final_error": final_error,
        "peak_speed": peak_speed,
        "peak_torque": peak_torque,
        "rms_error": rms_error,
        "control_effort": control_effort,
        "settling_time": settling_time
    }


# ============================================================
# Run both cases
# ============================================================

for case_name in CASES:
    run_case(case_name)


# ============================================================
# Terminal summary
# ============================================================

print("\n========== PD vs GRAVITY COMPENSATION ==========")

print(f"Joint        : joint2")
print(f"Target       : {TARGET:.3f} rad")
print(f"Kp           : {KP:.1f}")
print(f"Kd           : {KD:.1f}")

print(
    f"Initial gravity torque estimate: "
    f"{results['PD_only']['initial_gravity']:.6f} N·m"
)

print()

for name in CASES:

    r = results[name]

    print(f"--- {name} ---")

    print(
        f"Final q          : "
        f"{r['final_q']:.6f} rad"
    )

    print(
        f"Final error      : "
        f"{r['final_error']:.6f} rad"
    )

    print(
        f"RMS error        : "
        f"{r['rms_error']:.6f} rad"
    )

    print(
        f"Peak speed       : "
        f"{r['peak_speed']:.6f} rad/s"
    )

    print(
        f"Peak torque      : "
        f"{r['peak_torque']:.6f} N·m"
    )

    print(
        f"Control effort   : "
        f"{r['control_effort']:.6f} N²·m²·s"
    )

    print(
        f"Final gravity τ  : "
        f"{r['final_gravity']:.6f} N·m"
    )

    if np.isnan(r["settling_time"]):
        print(
            "Settling time    : "
            "NOT settled within ±0.01 rad"
        )
    else:
        print(
            f"Settling time    : "
            f"{r['settling_time']:.6f} s"
        )

    print()


# ============================================================
# Plot 1: Position comparison
# ============================================================

plt.figure(figsize=(8, 5))

for name in CASES:

    r = results[name]

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
plt.title("PD vs Gravity Compensation: Joint Position")

plt.legend()
plt.grid(True)
plt.tight_layout()

position_fig = os.path.join(
    OUTPUT_DIR,
    "pd_vs_gravity_position.png"
)

plt.savefig(
    position_fig,
    dpi=200
)


# ============================================================
# Plot 2: Torque decomposition
# ============================================================

r = results["PD_plus_gravity"]

plt.figure(figsize=(8, 5))

plt.plot(
    r["time"],
    r["tau_pd"],
    label="PD torque"
)

plt.plot(
    r["time"],
    r["tau_g"],
    label="Gravity compensation"
)

plt.plot(
    r["time"],
    r["tau_cmd"],
    label="Total command torque"
)

plt.axhline(
    0.0,
    linestyle="--"
)

plt.xlabel("Time (s)")
plt.ylabel("Torque (N·m)")
plt.title("PD + Gravity Compensation: Torque Components")

plt.legend()
plt.grid(True)
plt.tight_layout()

torque_fig = os.path.join(
    OUTPUT_DIR,
    "gravity_torque_components.png"
)

plt.savefig(
    torque_fig,
    dpi=200
)


print("[PASS] PD vs gravity compensation experiment finished")

print("\nFigures saved to:")
print(position_fig)
print(torque_fig)
