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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classical" / "inverse_dynamics"

os.makedirs(OUTPUT_DIR, exist_ok=True)

JOINT_ID = 1
ACTUATOR_ID = 1

# ============================================================
# Feedback gains
# Same gains are used in BOTH controllers.
# ============================================================

KP = 200.0
KD = 20.0

TORQUE_LIMIT = 87.0

SIM_TIME = 6.0

# ============================================================
# Desired sinusoidal trajectory
#
# qd(t)   = CENTER + AMP * sin(OMEGA*t)
# dqd(t)  = AMP * OMEGA * cos(OMEGA*t)
# ddqd(t) = -AMP * OMEGA^2 * sin(OMEGA*t)
# ============================================================

CENTER = -0.5
AMP = 0.25
FREQ = 0.5

OMEGA = 2.0 * np.pi * FREQ

CASES = [
    "PD_plus_gravity",
    "PD_plus_inverse_dynamics"
]

results = {}


# ============================================================
# Desired trajectory
# ============================================================

def desired_trajectory(t):

    qd = (
        CENTER
        + AMP * np.sin(OMEGA * t)
    )

    dqd = (
        AMP
        * OMEGA
        * np.cos(OMEGA * t)
    )

    ddqd = (
        -AMP
        * OMEGA**2
        * np.sin(OMEGA * t)
    )

    return qd, dqd, ddqd


# ============================================================
# Gravity-only torque
#
# Copy current configuration, but set velocity = 0.
# This removes velocity-dependent bias terms.
# ============================================================

def get_gravity_torque(model, helper, qpos):

    helper.qpos[:] = qpos
    helper.qvel[:] = 0.0
    helper.qacc[:] = 0.0

    mujoco.mj_forward(model, helper)

    return float(
        helper.qfrc_bias[JOINT_ID]
    )


# ============================================================
# Extract the full mass matrix M(q)
# ============================================================

def get_mass_matrix(model, data):

    M = np.zeros(
        (model.nv, model.nv),
        dtype=np.float64
    )

    # MuJoCo 3.11 Python API:
    # mj_fullM(model, data, destination)
    mujoco.mj_fullM(
        model,
        data,
        M
    )

    return M


# ============================================================
# Run one controller
# ============================================================

def run_case(case_name):

    model = mujoco.MjModel.from_xml_path(
        MODEL_PATH
    )

    data = mujoco.MjData(model)

    gravity_helper = mujoco.MjData(model)

    # --------------------------------------------------------
    # Convert joint2 actuator to direct torque control
    # --------------------------------------------------------

    model.actuator_gainprm[
        ACTUATOR_ID, :
    ] = 0.0

    model.actuator_gainprm[
        ACTUATOR_ID, 0
    ] = 1.0

    model.actuator_biasprm[
        ACTUATOR_ID, :
    ] = 0.0

    model.actuator_ctrlrange[
        ACTUATOR_ID
    ] = [
        -TORQUE_LIMIT,
        TORQUE_LIMIT
    ]

    model.actuator_ctrllimited[
        ACTUATOR_ID
    ] = 1

    model.actuator_forcerange[
        ACTUATOR_ID
    ] = [
        -TORQUE_LIMIT,
        TORQUE_LIMIT
    ]

    model.actuator_forcelimited[
        ACTUATOR_ID
    ] = 1

    # --------------------------------------------------------
    # Start exactly at qd(0)
    # --------------------------------------------------------

    qd0, dqd0, _ = desired_trajectory(0.0)

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    data.qpos[JOINT_ID] = qd0

    # Important:
    # start velocity at zero intentionally,
    # while desired velocity is nonzero.
    #
    # Therefore the controller must handle
    # a genuine initial dynamic tracking error.
    # --------------------------------------------------------

    mujoco.mj_forward(
        model,
        data
    )

    dt = model.opt.timestep
    steps = int(
        SIM_TIME / dt
    )

    records = []

    for _ in range(steps):

        t = float(data.time)

        qd, dqd, ddqd = desired_trajectory(t)

        q = float(
            data.qpos[JOINT_ID]
        )

        dq = float(
            data.qvel[JOINT_ID]
        )

        position_error = qd - q
        velocity_error = dqd - dq

        # ----------------------------------------------------
        # Same PD feedback for both cases
        # ----------------------------------------------------

        tau_pd = (
            KP * position_error
            + KD * velocity_error
        )

        # ----------------------------------------------------
        # CASE 1
        # PD + gravity compensation
        # ----------------------------------------------------

        tau_g = get_gravity_torque(
            model,
            gravity_helper,
            data.qpos
        )

        # ----------------------------------------------------
        # CASE 2
        # Inverse dynamics feedforward
        #
        # tau_ff =
        #   M(q) * qdd_des
        #   + h(q, qdot)
        #
        # qfrc_bias ~= h(q, qdot)
        # ----------------------------------------------------

        M = get_mass_matrix(
            model,
            data
        )

        desired_acceleration = np.zeros(
            model.nv
        )

        desired_acceleration[
            JOINT_ID
        ] = ddqd

        inertia_feedforward = float(
            M[JOINT_ID, :]
            @ desired_acceleration
        )

        dynamic_bias = float(
            data.qfrc_bias[JOINT_ID]
        )

        tau_inverse_dynamics = (
            inertia_feedforward
            + dynamic_bias
        )

        if case_name == "PD_plus_gravity":

            tau_ff = tau_g

        elif case_name == "PD_plus_inverse_dynamics":

            tau_ff = tau_inverse_dynamics

        else:
            raise ValueError(
                "Unknown controller case"
            )

        tau_cmd = (
            tau_pd
            + tau_ff
        )

        tau_cmd = float(
            np.clip(
                tau_cmd,
                -TORQUE_LIMIT,
                TORQUE_LIMIT
            )
        )

        data.ctrl[
            ACTUATOR_ID
        ] = tau_cmd

        # Other arm joints remain under
        # their original position servos.
        data.ctrl[0] = 0.0
        data.ctrl[2:7] = 0.0

        mujoco.mj_step(
            model,
            data
        )

        records.append([
            data.time,

            qd,
            dqd,
            ddqd,

            data.qpos[JOINT_ID],
            data.qvel[JOINT_ID],

            qd - data.qpos[JOINT_ID],

            tau_pd,
            tau_ff,
            tau_cmd,

            tau_g,
            inertia_feedforward,
            dynamic_bias
        ])

    arr = np.asarray(
        records
    )

    time = arr[:, 0]

    qd = arr[:, 1]
    dqd = arr[:, 2]
    ddqd = arr[:, 3]

    q = arr[:, 4]
    dq = arr[:, 5]

    error = arr[:, 6]

    tau_pd = arr[:, 7]
    tau_ff = arr[:, 8]
    tau_cmd = arr[:, 9]

    tau_g = arr[:, 10]
    inertia_ff = arr[:, 11]
    dynamic_bias = arr[:, 12]

    # ========================================================
    # Metrics
    # ========================================================

    rms_error = np.sqrt(
        np.mean(
            error**2
        )
    )

    max_abs_error = np.max(
        np.abs(
            error
        )
    )

    mean_abs_error = np.mean(
        np.abs(
            error
        )
    )

    peak_torque = np.max(
        np.abs(
            tau_cmd
        )
    )

    control_effort = np.sum(
        tau_cmd**2
    ) * dt

    results[case_name] = {
        "time": time,

        "qd": qd,
        "dqd": dqd,
        "ddqd": ddqd,

        "q": q,
        "dq": dq,

        "error": error,

        "tau_pd": tau_pd,
        "tau_ff": tau_ff,
        "tau_cmd": tau_cmd,

        "tau_g": tau_g,
        "inertia_ff": inertia_ff,
        "dynamic_bias": dynamic_bias,

        "rms_error": rms_error,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,

        "peak_torque": peak_torque,
        "control_effort": control_effort
    }

    # ========================================================
    # Save raw CSV
    # ========================================================

    csv_path = os.path.join(
        OUTPUT_DIR,
        f"{case_name}.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "time_s",

            "desired_position_rad",
            "desired_velocity_rad_s",
            "desired_acceleration_rad_s2",

            "actual_position_rad",
            "actual_velocity_rad_s",

            "position_error_rad",

            "pd_torque_Nm",
            "feedforward_torque_Nm",
            "command_torque_Nm",

            "gravity_torque_Nm",
            "inertia_feedforward_Nm",
            "dynamic_bias_Nm"
        ])

        writer.writerows(
            records
        )


# ============================================================
# Run experiments
# ============================================================

for case_name in CASES:

    run_case(
        case_name
    )


# ============================================================
# Terminal summary
# ============================================================

print(
    "\n========== DYNAMIC TRACKING COMPARISON =========="
)

print(
    f"Trajectory center : {CENTER:.3f} rad"
)

print(
    f"Amplitude         : {AMP:.3f} rad"
)

print(
    f"Frequency         : {FREQ:.3f} Hz"
)

print(
    f"Kp                : {KP:.1f}"
)

print(
    f"Kd                : {KD:.1f}"
)

print()

for name in CASES:

    r = results[name]

    print(
        f"--- {name} ---"
    )

    print(
        f"RMS error       : "
        f"{r['rms_error']:.6f} rad"
    )

    print(
        f"Mean abs error  : "
        f"{r['mean_abs_error']:.6f} rad"
    )

    print(
        f"Max abs error   : "
        f"{r['max_abs_error']:.6f} rad"
    )

    print(
        f"Peak torque     : "
        f"{r['peak_torque']:.6f} N·m"
    )

    print(
        f"Control effort  : "
        f"{r['control_effort']:.6f} N²·m²·s"
    )

    print()


# ============================================================
# Plot 1
# Desired vs actual position
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    results[
        "PD_plus_gravity"
    ]["time"],

    results[
        "PD_plus_gravity"
    ]["qd"],

    linestyle="--",
    label="Desired trajectory"
)

for name in CASES:

    r = results[name]

    plt.plot(
        r["time"],
        r["q"],
        label=name
    )

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Joint 2 Position (rad)"
)

plt.title(
    "Dynamic Tracking: Gravity vs Inverse Dynamics"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

position_fig = os.path.join(
    OUTPUT_DIR,
    "dynamic_tracking_position.png"
)

plt.savefig(
    position_fig,
    dpi=200
)


# ============================================================
# Plot 2
# Tracking error
# ============================================================

plt.figure(
    figsize=(9, 5)
)

for name in CASES:

    r = results[name]

    plt.plot(
        r["time"],
        r["error"],
        label=name
    )

plt.axhline(
    0.0,
    linestyle="--"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Position Error (rad)"
)

plt.title(
    "Dynamic Tracking Error"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

error_fig = os.path.join(
    OUTPUT_DIR,
    "dynamic_tracking_error.png"
)

plt.savefig(
    error_fig,
    dpi=200
)


# ============================================================
# Plot 3
# Feedforward decomposition
# ============================================================

r = results[
    "PD_plus_inverse_dynamics"
]

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    r["time"],
    r["tau_pd"],
    label="PD feedback"
)

plt.plot(
    r["time"],
    r["tau_ff"],
    label="Inverse-dynamics feedforward"
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

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Torque (N·m)"
)

plt.title(
    "Inverse-Dynamics Controller: Torque Components"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

torque_fig = os.path.join(
    OUTPUT_DIR,
    "inverse_dynamics_torque_components.png"
)

plt.savefig(
    torque_fig,
    dpi=200
)


print(
    "[PASS] inverse-dynamics comparison finished"
)

print(
    "\nFigures saved to:"
)

print(
    position_fig
)

print(
    error_fig
)

print(
    torque_fig
)
