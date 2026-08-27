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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classical" / "computed_torque"

os.makedirs(OUTPUT_DIR, exist_ok=True)

JOINT_ID = 1
ACTUATOR_ID = 1

# ============================================================
# Desired trajectory
# ============================================================

CENTER = -0.5
AMP = 0.25
FREQ = 0.5
OMEGA = 2.0 * np.pi * FREQ

SIM_TIME = 6.0
TORQUE_LIMIT = 87.0

# ============================================================
# Controller gains
#
# PD / inverse dynamics:
# torque-domain gains
#
# Computed torque:
# acceleration-domain gains
# ============================================================

KP_TORQUE = 200.0
KD_TORQUE = 20.0

KP_CT = 64.0
KD_CT = 16.0

CASES = [
    "PD_plus_gravity",
    "PD_plus_inverse_dynamics",
    "Computed_torque"
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
# ============================================================

def get_gravity_torque(model, helper, qpos):

    helper.qpos[:] = qpos
    helper.qvel[:] = 0.0
    helper.qacc[:] = 0.0

    mujoco.mj_forward(
        model,
        helper
    )

    return float(
        helper.qfrc_bias[JOINT_ID]
    )


# ============================================================
# Dense mass matrix
#
# IMPORTANT:
# MuJoCo 3.11.0 API used in the current environment:
#
# mj_fullM(model, data, M)
# ============================================================

def get_mass_matrix(model, data):

    M = np.zeros(
        (model.nv, model.nv),
        dtype=np.float64
    )

    mujoco.mj_fullM(
        model,
        data,
        M
    )

    return M


# ============================================================
# Configure joint2 for direct torque control
# ============================================================

def configure_torque_actuator(model):

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


# ============================================================
# Run one controller
# ============================================================

def run_case(case_name):

    model = mujoco.MjModel.from_xml_path(
        MODEL_PATH
    )

    configure_torque_actuator(
        model
    )

    data = mujoco.MjData(model)
    gravity_helper = mujoco.MjData(model)

    # --------------------------------------------------------
    # Same initial condition as previous experiment
    # --------------------------------------------------------

    qd0, _, _ = desired_trajectory(0.0)

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    data.qpos[
        JOINT_ID
    ] = qd0

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

        t = float(
            data.time
        )

        qd, dqd, ddqd = desired_trajectory(t)

        q = float(
            data.qpos[JOINT_ID]
        )

        dq = float(
            data.qvel[JOINT_ID]
        )

        e = qd - q
        edot = dqd - dq

        M = get_mass_matrix(
            model,
            data
        )

        # ----------------------------------------------------
        # Current nonlinear bias term
        #
        # ~= Coriolis + centrifugal + gravity
        # ----------------------------------------------------

        h = float(
            data.qfrc_bias[JOINT_ID]
        )

        gravity = get_gravity_torque(
            model,
            gravity_helper,
            data.qpos
        )

        # ----------------------------------------------------
        # Case 1:
        # PD + gravity
        # ----------------------------------------------------

        if case_name == "PD_plus_gravity":

            tau_feedback = (
                KP_TORQUE * e
                + KD_TORQUE * edot
            )

            tau_model = gravity

            tau_cmd = (
                tau_feedback
                + tau_model
            )

            virtual_acc = np.nan

        # ----------------------------------------------------
        # Case 2:
        # PD + inverse-dynamics feedforward
        # ----------------------------------------------------

        elif case_name == "PD_plus_inverse_dynamics":

            tau_feedback = (
                KP_TORQUE * e
                + KD_TORQUE * edot
            )

            desired_acc_vec = np.zeros(
                model.nv
            )

            desired_acc_vec[
                JOINT_ID
            ] = ddqd

            inertia_ff = float(
                M[JOINT_ID, :]
                @ desired_acc_vec
            )

            tau_model = (
                inertia_ff
                + h
            )

            tau_cmd = (
                tau_feedback
                + tau_model
            )

            virtual_acc = ddqd

        # ----------------------------------------------------
        # Case 3:
        # Computed torque
        #
        # qdd_virtual =
        #     qdd_des
        #     + Kp*e
        #     + Kd*edot
        #
        # tau =
        #     M(q) * qdd_virtual
        #     + h(q, qdot)
        # ----------------------------------------------------

        elif case_name == "Computed_torque":

            virtual_acc = (
                ddqd
                + KP_CT * e
                + KD_CT * edot
            )

            virtual_acc_vec = np.zeros(
                model.nv
            )

            virtual_acc_vec[
                JOINT_ID
            ] = virtual_acc

            tau_model = float(
                M[JOINT_ID, :]
                @ virtual_acc_vec
            )

            tau_feedback = (
                tau_model
                - float(
                    M[JOINT_ID, JOINT_ID]
                    * ddqd
                )
            )

            tau_cmd = (
                tau_model
                + h
            )

        else:

            raise ValueError(
                "Unknown controller"
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

        # Other joints retain original position servo
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

            tau_feedback,
            tau_model,
            tau_cmd,

            gravity,
            h,
            virtual_acc
        ])

    arr = np.asarray(
        records
    )

    time = arr[:, 0]
    qd = arr[:, 1]
    q = arr[:, 4]

    error = arr[:, 6]

    tau_feedback = arr[:, 7]
    tau_model = arr[:, 8]
    tau_cmd = arr[:, 9]

    # ========================================================
    # Metrics
    # ========================================================

    rms_error = np.sqrt(
        np.mean(
            error**2
        )
    )

    mean_abs_error = np.mean(
        np.abs(
            error
        )
    )

    max_abs_error = np.max(
        np.abs(
            error
        )
    )

    # Ignore first 1 s to evaluate periodic tracking
    mask_after_1s = (
        time >= 1.0
    )

    rms_error_after_1s = np.sqrt(
        np.mean(
            error[
                mask_after_1s
            ] ** 2
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

    results[
        case_name
    ] = {
        "time": time,
        "qd": qd,
        "q": q,
        "error": error,

        "tau_feedback": tau_feedback,
        "tau_model": tau_model,
        "tau_cmd": tau_cmd,

        "rms_error": rms_error,
        "rms_error_after_1s": rms_error_after_1s,
        "mean_abs_error": mean_abs_error,
        "max_abs_error": max_abs_error,

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

        writer = csv.writer(
            f
        )

        writer.writerow([
            "time_s",

            "desired_position_rad",
            "desired_velocity_rad_s",
            "desired_acceleration_rad_s2",

            "actual_position_rad",
            "actual_velocity_rad_s",

            "position_error_rad",

            "feedback_component",
            "model_component",
            "command_torque_Nm",

            "gravity_torque_Nm",
            "dynamic_bias_Nm",
            "virtual_acceleration_rad_s2"
        ])

        writer.writerows(
            records
        )


# ============================================================
# Run all cases
# ============================================================

for case_name in CASES:

    run_case(
        case_name
    )


# ============================================================
# Summary
# ============================================================

print(
    "\n========== COMPUTED TORQUE COMPARISON =========="
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

print()

print(
    f"Torque-PD gains   : "
    f"Kp={KP_TORQUE:.1f}, "
    f"Kd={KD_TORQUE:.1f}"
)

print(
    f"Computed-torque acceleration gains: "
    f"Kp={KP_CT:.1f}, "
    f"Kd={KD_CT:.1f}"
)

print()

for name in CASES:

    r = results[
        name
    ]

    print(
        f"--- {name} ---"
    )

    print(
        f"RMS error             : "
        f"{r['rms_error']:.6f} rad"
    )

    print(
        f"RMS error after 1 s   : "
        f"{r['rms_error_after_1s']:.6f} rad"
    )

    print(
        f"Mean abs error        : "
        f"{r['mean_abs_error']:.6f} rad"
    )

    print(
        f"Max abs error         : "
        f"{r['max_abs_error']:.6f} rad"
    )

    print(
        f"Peak torque           : "
        f"{r['peak_torque']:.6f} N·m"
    )

    print(
        f"Control effort        : "
        f"{r['control_effort']:.6f} N²·m²·s"
    )

    print()


# ============================================================
# Figure 1: position
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    results[
        "Computed_torque"
    ]["time"],

    results[
        "Computed_torque"
    ]["qd"],

    linestyle="--",
    label="Desired trajectory"
)

for name in CASES:

    r = results[
        name
    ]

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
    "Dynamic Tracking Controller Comparison"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

position_fig = os.path.join(
    OUTPUT_DIR,
    "computed_torque_position.png"
)

plt.savefig(
    position_fig,
    dpi=200
)


# ============================================================
# Figure 2: tracking error
# ============================================================

plt.figure(
    figsize=(9, 5)
)

for name in CASES:

    r = results[
        name
    ]

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
    "Tracking Error Comparison"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

error_fig = os.path.join(
    OUTPUT_DIR,
    "computed_torque_error.png"
)

plt.savefig(
    error_fig,
    dpi=200
)


# ============================================================
# Figure 3: Computed torque decomposition
# ============================================================

r = results[
    "Computed_torque"
]

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    r["time"],
    r["tau_feedback"],
    label="Feedback-related torque"
)

plt.plot(
    r["time"],
    r["tau_model"],
    label="Model-based acceleration torque"
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
    "Computed Torque: Control Components"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

components_fig = os.path.join(
    OUTPUT_DIR,
    "computed_torque_components.png"
)

plt.savefig(
    components_fig,
    dpi=200
)


print(
    "[PASS] computed-torque comparison finished"
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
    components_fig
)
