import os
import csv
import time
from pathlib import Path
from collections import Counter

import numpy as np
import mujoco
import cvxpy as cp
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MENAGERIE_ROOT = Path(os.environ.get(
    "MUJOCO_MENAGERIE_PATH",
    PROJECT_ROOT / "third_party" / "mujoco_menagerie",
)).expanduser()
MODEL_PATH = str(MENAGERIE_ROOT / "franka_emika_panda" / "scene.xml")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mpc" / "constraint_challenge"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Robot / limits
# ============================================================

JOINT_ID = 1
ACTUATOR_ID = 1

TORQUE_LIMIT = 35.0

MAX_TORQUE_STEP = 8.0


# ============================================================
# Timing
# ============================================================

SIM_TIME = 6.0

CONTROL_DT = 0.020

HORIZON_STEPS = 10

PREDICTION_TIME = (
    CONTROL_DT * HORIZON_STEPS
)


# ============================================================
# Faster cosine trajectory
#
# Start with zero desired velocity:
#
# qd(0) = CENTER + AMP
# dqd(0) = 0
# ============================================================

CENTER = -0.5
AMP = 0.20
FREQ = 0.75

OMEGA = 2.0 * np.pi * FREQ


def desired_trajectory(t):

    qd = (
        CENTER
        + AMP * np.cos(OMEGA * t)
    )

    dqd = (
        -AMP
        * OMEGA
        * np.sin(OMEGA * t)
    )

    ddqd = (
        -AMP
        * OMEGA**2
        * np.cos(OMEGA * t)
    )

    return qd, dqd, ddqd


# ============================================================
# Computed-torque gains
# ============================================================

KP_CT = 64.0
KD_CT = 16.0


# ============================================================
# MPC normalization / weights
# ============================================================

POSITION_ERROR_SCALE = 0.05
VELOCITY_ERROR_SCALE = 0.50
CORRECTION_TORQUE_SCALE = 15.0

W_POSITION = 1.0
W_VELOCITY = 0.25

W_CORRECTION = 0.02
W_TORQUE_CHANGE = 0.02

W_TERMINAL_POSITION = 5.0
W_TERMINAL_VELOCITY = 1.0


# ============================================================
# MuJoCo 3.11.0 mass matrix
#
# Correct API:
#
# mujoco.mj_fullM(model, data, M)
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
# Direct torque actuator
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
# Initial model/data
# ============================================================

def create_simulation():

    model = mujoco.MjModel.from_xml_path(
        MODEL_PATH
    )

    configure_torque_actuator(
        model
    )

    data = mujoco.MjData(
        model
    )

    qd0, _, _ = desired_trajectory(
        0.0
    )

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    data.qpos[
        JOINT_ID
    ] = qd0

    mujoco.mj_forward(
        model,
        data
    )

    return model, data


# ============================================================
# Safety constraint helper
#
# This is used for Computed Torque AFTER it has
# already generated an unconstrained command.
# ============================================================

def posthoc_safety_clip(
    raw_torque,
    previous_torque
):

    magnitude_clipped = float(
        np.clip(
            raw_torque,
            -TORQUE_LIMIT,
            TORQUE_LIMIT
        )
    )

    final_torque = float(
        np.clip(
            magnitude_clipped,

            previous_torque
            - MAX_TORQUE_STEP,

            previous_torque
            + MAX_TORQUE_STEP
        )
    )

    return (
        final_torque,
        magnitude_clipped
    )


# ============================================================
# CASE 1:
# Computed Torque + post-hoc clipping
# ============================================================

def run_computed_torque():

    model, data = create_simulation()

    physics_dt = float(
        model.opt.timestep
    )

    physics_steps_per_control = int(
        round(
            CONTROL_DT / physics_dt
        )
    )

    initial_bias = float(
        data.qfrc_bias[
            JOINT_ID
        ]
    )

    previous_torque = float(
        np.clip(
            initial_bias,
            -TORQUE_LIMIT,
            TORQUE_LIMIT
        )
    )

    records = []

    compute_times = []

    magnitude_clip_count = 0
    rate_clip_count = 0

    updates = int(
        SIM_TIME / CONTROL_DT
    )

    for _ in range(updates):

        tic = time.perf_counter()

        t = float(
            data.time
        )

        qd, dqd, ddqd = desired_trajectory(
            t
        )

        q = float(
            data.qpos[
                JOINT_ID
            ]
        )

        dq = float(
            data.qvel[
                JOINT_ID
            ]
        )

        e = qd - q
        edot = dqd - dq

        M = get_mass_matrix(
            model,
            data
        )

        inertia = float(
            M[
                JOINT_ID,
                JOINT_ID
            ]
        )

        h = float(
            data.qfrc_bias[
                JOINT_ID
            ]
        )

        virtual_acc = (
            ddqd
            + KP_CT * e
            + KD_CT * edot
        )

        raw_torque = (
            inertia
            * virtual_acc
            + h
        )

        torque_command, magnitude_only = (
            posthoc_safety_clip(
                raw_torque,
                previous_torque
            )
        )

        if abs(
            magnitude_only
            - raw_torque
        ) > 1e-8:

            magnitude_clip_count += 1

        if abs(
            torque_command
            - magnitude_only
        ) > 1e-8:

            rate_clip_count += 1

        compute_times.append(
            time.perf_counter()
            - tic
        )

        data.ctrl[
            ACTUATOR_ID
        ] = torque_command

        data.ctrl[0] = 0.0
        data.ctrl[2:7] = 0.0

        for _ in range(
            physics_steps_per_control
        ):

            mujoco.mj_step(
                model,
                data
            )

        qd_now, dqd_now, _ = desired_trajectory(
            data.time
        )

        actual_q = float(
            data.qpos[
                JOINT_ID
            ]
        )

        actual_dq = float(
            data.qvel[
                JOINT_ID
            ]
        )

        records.append([
            data.time,

            qd_now,
            dqd_now,

            actual_q,
            actual_dq,

            actual_q - qd_now,

            raw_torque,
            torque_command,

            inertia,
            h
        ])

        previous_torque = torque_command

    return {
        "records": records,

        "compute_times":
            np.asarray(
                compute_times
            ),

        "magnitude_clip_count":
            magnitude_clip_count,

        "rate_clip_count":
            rate_clip_count
    }


# ============================================================
# MPC problem
#
# Error state:
#
# e     = q - qd
# edot  = dq - dqd
#
# nominal:
#
# tau_ff = M*qdd_des + h
#
# MPC optimizes correction delta_tau.
# ============================================================

N = HORIZON_STEPS

z = cp.Variable(
    (2, N + 1)
)

delta_tau = cp.Variable(
    N
)

z0_param = cp.Parameter(
    2
)

bq_param = cp.Parameter(
    nonneg=True
)

bv_param = cp.Parameter(
    nonneg=True
)

tau_ff_param = cp.Parameter(
    N
)

previous_tau_param = cp.Parameter()


constraints = [
    z[:, 0] == z0_param
]

cost = 0.0


for k in range(N):

    pos_norm = (
        z[0, k]
        / POSITION_ERROR_SCALE
    )

    vel_norm = (
        z[1, k]
        / VELOCITY_ERROR_SCALE
    )

    correction_norm = (
        delta_tau[k]
        / CORRECTION_TORQUE_SCALE
    )

    cost += (
        W_POSITION
        * cp.square(
            pos_norm
        )
    )

    cost += (
        W_VELOCITY
        * cp.square(
            vel_norm
        )
    )

    cost += (
        W_CORRECTION
        * cp.square(
            correction_norm
        )
    )

    tau_k = (
        tau_ff_param[k]
        + delta_tau[k]
    )

    if k == 0:

        torque_change = (
            tau_k
            - previous_tau_param
        )

    else:

        tau_previous = (
            tau_ff_param[k - 1]
            + delta_tau[k - 1]
        )

        torque_change = (
            tau_k
            - tau_previous
        )

    cost += (
        W_TORQUE_CHANGE
        * cp.square(
            torque_change
            / MAX_TORQUE_STEP
        )
    )

    # Error dynamics

    constraints += [

        z[0, k + 1]
        ==
        z[0, k]
        + CONTROL_DT
        * z[1, k]
        + bq_param
        * delta_tau[k],

        z[1, k + 1]
        ==
        z[1, k]
        + bv_param
        * delta_tau[k]
    ]

    # Hard torque constraint

    constraints += [

        tau_k
        <= TORQUE_LIMIT,

        tau_k
        >= -TORQUE_LIMIT
    ]

    # Hard rate constraint

    constraints += [

        torque_change
        <= MAX_TORQUE_STEP,

        torque_change
        >= -MAX_TORQUE_STEP
    ]


cost += (

    W_TERMINAL_POSITION
    * cp.square(
        z[0, N]
        / POSITION_ERROR_SCALE
    )
)

cost += (

    W_TERMINAL_VELOCITY
    * cp.square(
        z[1, N]
        / VELOCITY_ERROR_SCALE
    )
)


mpc_problem = cp.Problem(
    cp.Minimize(cost),
    constraints
)


# ============================================================
# CASE 2:
# Constrained MPC
# ============================================================

def run_mpc():

    model, data = create_simulation()

    physics_dt = float(
        model.opt.timestep
    )

    physics_steps_per_control = int(
        round(
            CONTROL_DT / physics_dt
        )
    )

    initial_bias = float(
        data.qfrc_bias[
            JOINT_ID
        ]
    )

    previous_torque = float(
        np.clip(
            initial_bias,
            -TORQUE_LIMIT,
            TORQUE_LIMIT
        )
    )

    status_counter = Counter()

    solver_failures = 0

    safety_adjustment_count = 0

    records = []

    solve_times = []

    # --------------------------------------------------------
    # Parameter update function
    # --------------------------------------------------------

    def update_parameters(
        current_time,
        q,
        dq,
        inertia,
        dynamic_bias,
        previous_tau
    ):

        qd, dqd, _ = desired_trajectory(
            current_time
        )

        z0_param.value = np.array(
            [
                q - qd,
                dq - dqd
            ],
            dtype=np.float64
        )

        bq_param.value = (
            0.5
            * CONTROL_DT**2
            / inertia
        )

        bv_param.value = (
            CONTROL_DT
            / inertia
        )

        tau_ff = np.zeros(
            N,
            dtype=np.float64
        )

        for k in range(N):

            future_t = (
                current_time
                + k * CONTROL_DT
            )

            _, _, ddqd = desired_trajectory(
                future_t
            )

            tau_ff[k] = (
                inertia
                * ddqd
                + dynamic_bias
            )

        tau_ff_param.value = tau_ff

        previous_tau_param.value = (
            previous_tau
        )

        return tau_ff

    # --------------------------------------------------------
    # Warm-up solve
    # --------------------------------------------------------

    M0 = get_mass_matrix(
        model,
        data
    )

    inertia0 = float(
        M0[
            JOINT_ID,
            JOINT_ID
        ]
    )

    update_parameters(
        0.0,

        float(
            data.qpos[
                JOINT_ID
            ]
        ),

        float(
            data.qvel[
                JOINT_ID
            ]
        ),

        inertia0,

        initial_bias,

        previous_torque
    )

    mpc_problem.solve(
        solver=cp.OSQP,
        warm_start=True,
        verbose=False,
        eps_abs=1e-3,
        eps_rel=1e-3,
        max_iter=6000,
        polishing=False
    )

    # --------------------------------------------------------
    # Closed loop
    # --------------------------------------------------------

    updates = int(
        SIM_TIME / CONTROL_DT
    )

    for _ in range(updates):

        t = float(
            data.time
        )

        q = float(
            data.qpos[
                JOINT_ID
            ]
        )

        dq = float(
            data.qvel[
                JOINT_ID
            ]
        )

        M = get_mass_matrix(
            model,
            data
        )

        inertia = float(
            M[
                JOINT_ID,
                JOINT_ID
            ]
        )

        dynamic_bias = float(
            data.qfrc_bias[
                JOINT_ID
            ]
        )

        tau_ff = update_parameters(
            t,
            q,
            dq,
            inertia,
            dynamic_bias,
            previous_torque
        )

        tic = time.perf_counter()

        solve_ok = False
        status_string = "unknown"

        try:

            mpc_problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                eps_abs=1e-3,
                eps_rel=1e-3,
                max_iter=6000,
                polishing=False
            )

            status_string = str(
                mpc_problem.status
            )

            if (
                mpc_problem.status
                in [
                    cp.OPTIMAL,
                    cp.OPTIMAL_INACCURATE
                ]
                and
                delta_tau.value
                is not None
            ):

                solve_ok = True

        except Exception as exc:

            status_string = (
                "exception:"
                + type(exc).__name__
            )

        solve_time = (
            time.perf_counter()
            - tic
        )

        solve_times.append(
            solve_time
        )

        status_counter[
            status_string
        ] += 1

        nominal_torque = float(
            tau_ff[0]
        )

        if solve_ok:

            correction_torque = float(
                delta_tau.value[0]
            )

            optimized_torque = (
                nominal_torque
                + correction_torque
            )

        else:

            solver_failures += 1

            correction_torque = 0.0

            optimized_torque = nominal_torque

        # ----------------------------------------------------
        # Safety verification only.
        #
        # If MPC did its job correctly, this should make
        # essentially NO adjustment.
        # ----------------------------------------------------

        verified_torque = float(
            np.clip(
                optimized_torque,
                -TORQUE_LIMIT,
                TORQUE_LIMIT
            )
        )

        verified_torque = float(
            np.clip(
                verified_torque,

                previous_torque
                - MAX_TORQUE_STEP,

                previous_torque
                + MAX_TORQUE_STEP
            )
        )

        if abs(
            verified_torque
            - optimized_torque
        ) > 1e-6:

            safety_adjustment_count += 1

        torque_command = (
            verified_torque
        )

        data.ctrl[
            ACTUATOR_ID
        ] = torque_command

        data.ctrl[0] = 0.0
        data.ctrl[2:7] = 0.0

        for _ in range(
            physics_steps_per_control
        ):

            mujoco.mj_step(
                model,
                data
            )

        qd_now, dqd_now, _ = desired_trajectory(
            data.time
        )

        actual_q = float(
            data.qpos[
                JOINT_ID
            ]
        )

        actual_dq = float(
            data.qvel[
                JOINT_ID
            ]
        )

        records.append([
            data.time,

            qd_now,
            dqd_now,

            actual_q,
            actual_dq,

            actual_q - qd_now,

            nominal_torque,
            correction_torque,
            torque_command,

            inertia,
            dynamic_bias,

            solve_time
        ])

        previous_torque = torque_command

    return {
        "records": records,

        "solve_times":
            np.asarray(
                solve_times
            ),

        "solver_failures":
            solver_failures,

        "status_counter":
            status_counter,

        "safety_adjustment_count":
            safety_adjustment_count
    }


# ============================================================
# Run both
# ============================================================

print(
    "\n========== CONSTRAINT CHALLENGE =========="
)

print(
    f"Trajectory frequency  : "
    f"{FREQ:.2f} Hz"
)

print(
    f"Torque limit          : "
    f"±{TORQUE_LIMIT:.1f} N·m"
)

print(
    f"Torque-rate limit     : "
    f"±{MAX_TORQUE_STEP:.1f} N·m / step"
)

print(
    f"Control period        : "
    f"{CONTROL_DT * 1000:.1f} ms"
)

print(
    f"MPC horizon           : "
    f"{PREDICTION_TIME:.3f} s"
)

print()


ct_result = run_computed_torque()

mpc_result = run_mpc()


# ============================================================
# Convert / analyze helper
# ============================================================

def analyze_records(
    records,
    actual_torque_column
):

    arr = np.asarray(
        records,
        dtype=np.float64
    )

    time_arr = arr[:, 0]

    qd = arr[:, 1]
    q = arr[:, 3]

    error = arr[:, 5]

    torque = arr[
        :,
        actual_torque_column
    ]

    rms = float(
        np.sqrt(
            np.mean(
                error**2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(
                error
            )
        )
    )

    max_error = float(
        np.max(
            np.abs(
                error
            )
        )
    )

    peak_torque = float(
        np.max(
            np.abs(
                torque
            )
        )
    )

    effort = float(
        np.sum(
            torque**2
        )
        * CONTROL_DT
    )

    torque_steps = np.abs(
        np.diff(
            torque
        )
    )

    max_torque_step = float(
        np.max(
            torque_steps
        )
    )

    torque_limit_fraction = float(
        np.mean(
            np.abs(torque)
            >= TORQUE_LIMIT - 1e-3
        )
        * 100.0
    )

    rate_limit_fraction = float(
        np.mean(
            torque_steps
            >= MAX_TORQUE_STEP - 1e-3
        )
        * 100.0
    )

    return {
        "arr": arr,

        "time": time_arr,
        "qd": qd,
        "q": q,
        "error": error,
        "torque": torque,

        "rms": rms,
        "mae": mae,
        "max_error": max_error,

        "peak_torque": peak_torque,
        "effort": effort,

        "max_torque_step":
            max_torque_step,

        "torque_limit_fraction":
            torque_limit_fraction,

        "rate_limit_fraction":
            rate_limit_fraction
    }


ct = analyze_records(
    ct_result["records"],
    actual_torque_column=7
)

mpc = analyze_records(
    mpc_result["records"],
    actual_torque_column=8
)


# ============================================================
# Computed Torque raw violation analysis
# ============================================================

ct_raw = ct["arr"][:, 6]

raw_torque_violation_rate = float(
    np.mean(
        np.abs(
            ct_raw
        )
        > TORQUE_LIMIT
    )
    * 100.0
)


# ============================================================
# Terminal summary
# ============================================================

print(
    "========== COMPUTED TORQUE + CLIPPING =========="
)

print(
    f"RMS tracking error       : "
    f"{ct['rms']:.6f} rad"
)

print(
    f"Mean abs error           : "
    f"{ct['mae']:.6f} rad"
)

print(
    f"Max abs error            : "
    f"{ct['max_error']:.6f} rad"
)

print(
    f"Peak actual torque       : "
    f"{ct['peak_torque']:.6f} N·m"
)

print(
    f"Control effort           : "
    f"{ct['effort']:.6f} N²·m²·s"
)

print(
    f"Raw torque > limit       : "
    f"{raw_torque_violation_rate:.2f}%"
)

print(
    f"Magnitude clipping count : "
    f"{ct_result['magnitude_clip_count']}"
)

print(
    f"Rate clipping count      : "
    f"{ct_result['rate_clip_count']}"
)

print(
    f"Max actual torque step   : "
    f"{ct['max_torque_step']:.6f} N·m"
)

print(
    f"Torque-limit active      : "
    f"{ct['torque_limit_fraction']:.2f}%"
)

print(
    f"Rate-limit active        : "
    f"{ct['rate_limit_fraction']:.2f}%"
)

print(
    f"Average compute time     : "
    f"{np.mean(ct_result['compute_times']) * 1000:.4f} ms"
)

print()


print(
    "========== CONSTRAINED MPC =========="
)

print(
    f"RMS tracking error       : "
    f"{mpc['rms']:.6f} rad"
)

print(
    f"Mean abs error           : "
    f"{mpc['mae']:.6f} rad"
)

print(
    f"Max abs error            : "
    f"{mpc['max_error']:.6f} rad"
)

print(
    f"Peak actual torque       : "
    f"{mpc['peak_torque']:.6f} N·m"
)

print(
    f"Control effort           : "
    f"{mpc['effort']:.6f} N²·m²·s"
)

print(
    f"Max actual torque step   : "
    f"{mpc['max_torque_step']:.6f} N·m"
)

print(
    f"Torque-limit active      : "
    f"{mpc['torque_limit_fraction']:.2f}%"
)

print(
    f"Rate-limit active        : "
    f"{mpc['rate_limit_fraction']:.2f}%"
)

print(
    f"Safety adjustments       : "
    f"{mpc_result['safety_adjustment_count']}"
)

print(
    f"Solver failures          : "
    f"{mpc_result['solver_failures']}"
)

print(
    f"Solver status counts     : "
    f"{dict(mpc_result['status_counter'])}"
)

print(
    f"Average solver time      : "
    f"{np.mean(mpc_result['solve_times']) * 1000:.3f} ms"
)

print(
    f"95% solver time          : "
    f"{np.percentile(mpc_result['solve_times'], 95) * 1000:.3f} ms"
)

print(
    f"Deadline miss rate       : "
    f"{np.mean(mpc_result['solve_times'] > CONTROL_DT) * 100:.2f}%"
)


# ============================================================
# Save summary CSV
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "constraint_challenge_summary.csv"
)

with open(
    summary_path,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "controller",
        "rms_error_rad",
        "mean_abs_error_rad",
        "max_abs_error_rad",
        "peak_torque_Nm",
        "control_effort",
        "max_torque_step_Nm"
    ])

    writer.writerow([
        "ComputedTorque_clipped",
        ct["rms"],
        ct["mae"],
        ct["max_error"],
        ct["peak_torque"],
        ct["effort"],
        ct["max_torque_step"]
    ])

    writer.writerow([
        "Constrained_MPC",
        mpc["rms"],
        mpc["mae"],
        mpc["max_error"],
        mpc["peak_torque"],
        mpc["effort"],
        mpc["max_torque_step"]
    ])


# ============================================================
# Figure 1: tracking comparison
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    ct["time"],
    ct["qd"],
    linestyle="--",
    label="Desired trajectory"
)

plt.plot(
    ct["time"],
    ct["q"],
    label="Computed Torque + clipping"
)

plt.plot(
    mpc["time"],
    mpc["q"],
    label="Constrained MPC"
)

plt.xlabel("Time (s)")
plt.ylabel("Joint 2 Position (rad)")

plt.title(
    "Constraint Challenge: Tracking Comparison"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

position_fig = os.path.join(
    OUTPUT_DIR,
    "constraint_tracking_comparison.png"
)

plt.savefig(
    position_fig,
    dpi=200
)


# ============================================================
# Figure 2: error
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    ct["time"],
    ct["error"],
    label="Computed Torque + clipping"
)

plt.plot(
    mpc["time"],
    mpc["error"],
    label="Constrained MPC"
)

plt.axhline(
    0.0,
    linestyle="--"
)

plt.xlabel("Time (s)")
plt.ylabel("Position Error (rad)")

plt.title(
    "Constraint Challenge: Tracking Error"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

error_fig = os.path.join(
    OUTPUT_DIR,
    "constraint_error_comparison.png"
)

plt.savefig(
    error_fig,
    dpi=200
)


# ============================================================
# Figure 3: torque
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    ct["time"],
    ct_raw,
    linestyle=":",
    label="Computed Torque raw request"
)

plt.plot(
    ct["time"],
    ct["torque"],
    label="Computed Torque after clipping"
)

plt.plot(
    mpc["time"],
    mpc["torque"],
    label="Constrained MPC"
)

plt.axhline(
    TORQUE_LIMIT,
    linestyle="--",
    label="Torque limit"
)

plt.axhline(
    -TORQUE_LIMIT,
    linestyle="--"
)

plt.xlabel("Time (s)")
plt.ylabel("Torque (N·m)")

plt.title(
    "Constraint Challenge: Torque Commands"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

torque_fig = os.path.join(
    OUTPUT_DIR,
    "constraint_torque_comparison.png"
)

plt.savefig(
    torque_fig,
    dpi=200
)


print(
    "\n[PASS] constraint challenge finished"
)

print(
    "\nFigures saved to:"
)

print(position_fig)
print(error_fig)
print(torque_fig)

print(
    "\nSummary saved to:"
)

print(summary_path)
