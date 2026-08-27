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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mpc" / "nominal"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Robot
# ============================================================

JOINT_ID = 1
ACTUATOR_ID = 1

TORQUE_LIMIT = 87.0

# Maximum torque change per MPC control period
MAX_TORQUE_STEP = 15.0


# ============================================================
# Timing
# ============================================================

SIM_TIME = 6.0

CONTROL_DT = 0.020

HORIZON_STEPS = 10

PREDICTION_TIME = (
    CONTROL_DT
    * HORIZON_STEPS
)


# ============================================================
# Desired trajectory
# ============================================================

CENTER = -0.5
AMP = 0.25
FREQ = 0.5

OMEGA = (
    2.0
    * np.pi
    * FREQ
)


def desired_trajectory(t):

    qd = (
        CENTER
        + AMP
        * np.sin(
            OMEGA * t
        )
    )

    dqd = (
        AMP
        * OMEGA
        * np.cos(
            OMEGA * t
        )
    )

    ddqd = (
        -AMP
        * OMEGA**2
        * np.sin(
            OMEGA * t
        )
    )

    return qd, dqd, ddqd


# ============================================================
# Normalization scales
#
# "How large is one typical unit?"
# ============================================================

POSITION_ERROR_SCALE = 0.05      # rad
VELOCITY_ERROR_SCALE = 0.50      # rad/s
CORRECTION_TORQUE_SCALE = 20.0   # N·m


# ============================================================
# Dimensionless MPC weights
# ============================================================

W_POSITION = 1.0
W_VELOCITY = 0.25

W_CORRECTION_TORQUE = 0.02
W_TORQUE_CHANGE = 0.02

W_TERMINAL_POSITION = 5.0
W_TERMINAL_VELOCITY = 1.0


# ============================================================
# MuJoCo mass matrix
#
# IMPORTANT:
#
# Current environment:
# MuJoCo 3.11.0
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
# Torque actuator
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
# MPC formulation
#
# Error state:
#
# z = [e, edot]
#
# e    = q - qd
# edot = dq - dqd
#
# Nominal control:
#
# tau_ff = M * qdd_des + h
#
# MPC correction:
#
# delta_tau
#
# Total control:
#
# tau = tau_ff + delta_tau
#
# Approximate error dynamics:
#
# e_ddot = delta_tau / M
# ============================================================

N = HORIZON_STEPS


z = cp.Variable(
    (2, N + 1)
)

delta_tau = cp.Variable(
    N
)


# ============================================================
# Parameters
# ============================================================

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


# ============================================================
# Build optimization problem
# ============================================================

constraints = [

    z[:, 0]
    ==
    z0_param
]


cost = 0.0


for k in range(N):

    # --------------------------------------------------------
    # Normalized error cost
    # --------------------------------------------------------

    normalized_position_error = (
        z[0, k]
        / POSITION_ERROR_SCALE
    )

    normalized_velocity_error = (
        z[1, k]
        / VELOCITY_ERROR_SCALE
    )

    normalized_correction = (
        delta_tau[k]
        / CORRECTION_TORQUE_SCALE
    )


    cost += (
        W_POSITION
        * cp.square(
            normalized_position_error
        )
    )


    cost += (
        W_VELOCITY
        * cp.square(
            normalized_velocity_error
        )
    )


    cost += (
        W_CORRECTION_TORQUE
        * cp.square(
            normalized_correction
        )
    )


    # --------------------------------------------------------
    # Actual torque
    # --------------------------------------------------------

    tau_k = (
        tau_ff_param[k]
        + delta_tau[k]
    )


    # --------------------------------------------------------
    # Torque change cost
    # --------------------------------------------------------

    if k == 0:

        delta_command = (
            tau_k
            - previous_tau_param
        )

    else:

        tau_previous = (
            tau_ff_param[k - 1]
            + delta_tau[k - 1]
        )

        delta_command = (
            tau_k
            - tau_previous
        )


    cost += (
        W_TORQUE_CHANGE
        * cp.square(
            delta_command
            / MAX_TORQUE_STEP
        )
    )


    # --------------------------------------------------------
    # Error-state dynamics
    #
    # e_next =
    #     e
    #     + dt*edot
    #     + 0.5*dt^2/M * delta_tau
    #
    # edot_next =
    #     edot
    #     + dt/M * delta_tau
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Hard total-torque constraint
    # --------------------------------------------------------

    constraints += [

        tau_k
        <= TORQUE_LIMIT,

        tau_k
        >= -TORQUE_LIMIT
    ]


    # --------------------------------------------------------
    # Hard torque-rate constraint
    #
    # |tau_k - tau_previous|
    # <= MAX_TORQUE_STEP
    # --------------------------------------------------------

    constraints += [

        delta_command
        <= MAX_TORQUE_STEP,

        delta_command
        >= -MAX_TORQUE_STEP
    ]


# ============================================================
# Terminal cost
# ============================================================

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


problem = cp.Problem(

    cp.Minimize(
        cost
    ),

    constraints
)


# ============================================================
# Initialize MuJoCo
# ============================================================

model = mujoco.MjModel.from_xml_path(
    MODEL_PATH
)

configure_torque_actuator(
    model
)

data = mujoco.MjData(
    model
)


# Same initial condition as previous experiments

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


physics_dt = float(
    model.opt.timestep
)


physics_steps_per_control = int(

    round(
        CONTROL_DT
        / physics_dt
    )
)


# ============================================================
# Header
# ============================================================

print(
    "\n========== MPC V3: ERROR-STATE MPC =========="
)

print(
    f"MuJoCo version          : "
    f"{mujoco.__version__}"
)

print(
    f"Physics timestep        : "
    f"{physics_dt:.4f} s"
)

print(
    f"Control timestep        : "
    f"{CONTROL_DT:.4f} s"
)

print(
    f"Horizon steps           : "
    f"{HORIZON_STEPS}"
)

print(
    f"Prediction horizon      : "
    f"{PREDICTION_TIME:.3f} s"
)

print(
    f"Torque limit            : "
    f"±{TORQUE_LIMIT:.1f} N·m"
)

print(
    f"Max torque step         : "
    f"{MAX_TORQUE_STEP:.1f} N·m / control step"
)

print(
    f"CVXPY DPP compliant     : "
    f"{problem.is_dpp()}"
)


# ============================================================
# Helper:
# update MPC parameters
# ============================================================

def update_mpc_parameters(
    current_time,
    q,
    dq,
    inertia,
    dynamic_bias,
    previous_tau
):

    # --------------------------------------------------------
    # Current tracking error
    # --------------------------------------------------------

    qd, dqd, _ = desired_trajectory(
        current_time
    )

    position_error = (
        q - qd
    )

    velocity_error = (
        dq - dqd
    )


    z0_param.value = np.array(
        [
            position_error,
            velocity_error
        ],
        dtype=np.float64
    )


    # --------------------------------------------------------
    # Error-state discrete dynamics
    # --------------------------------------------------------

    bq_param.value = (
        0.5
        * CONTROL_DT**2
        / inertia
    )

    bv_param.value = (
        CONTROL_DT
        / inertia
    )


    # --------------------------------------------------------
    # Future nominal inverse-dynamics feedforward
    #
    # Frozen inertia / bias over short horizon
    # --------------------------------------------------------

    tau_ff = np.zeros(
        N,
        dtype=np.float64
    )


    for k in range(N):

        future_time = (
            current_time
            + k
            * CONTROL_DT
        )

        _, _, ddqd = desired_trajectory(
            future_time
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


# ============================================================
# Warm-up solve
# ============================================================

M0 = get_mass_matrix(
    model,
    data
)


initial_inertia = max(

    float(
        M0[
            JOINT_ID,
            JOINT_ID
        ]
    ),

    1e-6
)


initial_bias = float(

    data.qfrc_bias[
        JOINT_ID
    ]
)


update_mpc_parameters(

    current_time=0.0,

    q=float(
        data.qpos[
            JOINT_ID
        ]
    ),

    dq=float(
        data.qvel[
            JOINT_ID
        ]
    ),

    inertia=initial_inertia,

    dynamic_bias=initial_bias,

    previous_tau=initial_bias
)


warm_tic = time.perf_counter()


problem.solve(

    solver=cp.OSQP,

    warm_start=True,

    verbose=False,

    eps_abs=1e-3,

    eps_rel=1e-3,

    max_iter=6000,

    polishing=False
)


warmup_ms = (

    time.perf_counter()
    - warm_tic

) * 1000.0


print(
    f"Warm-up solve           : "
    f"{warmup_ms:.3f} ms "
    "(excluded)"
)


# ============================================================
# Receding-horizon closed loop
# ============================================================

records = []

previous_torque = initial_bias

solver_failures = 0

status_counter = Counter()


number_of_updates = int(

    SIM_TIME
    / CONTROL_DT
)


for _ in range(
    number_of_updates
):

    current_time = float(
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


    # --------------------------------------------------------
    # Update real nonlinear dynamics
    # --------------------------------------------------------

    M = get_mass_matrix(
        model,
        data
    )


    inertia = max(

        float(
            M[
                JOINT_ID,
                JOINT_ID
            ]
        ),

        1e-6
    )


    dynamic_bias = float(

        data.qfrc_bias[
            JOINT_ID
        ]
    )


    tau_ff_sequence = update_mpc_parameters(

        current_time,
        q,
        dq,
        inertia,
        dynamic_bias,
        previous_torque
    )


    # --------------------------------------------------------
    # Online QP solve
    # --------------------------------------------------------

    tic = time.perf_counter()

    status_string = "unknown"

    solve_ok = False


    try:

        problem.solve(

            solver=cp.OSQP,

            warm_start=True,

            verbose=False,

            eps_abs=1e-3,

            eps_rel=1e-3,

            max_iter=6000,

            polishing=False
        )


        status_string = str(
            problem.status
        )


        if (

            problem.status
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


    status_counter[
        status_string
    ] += 1


    # --------------------------------------------------------
    # Receding horizon:
    # execute ONLY first action
    # --------------------------------------------------------

    nominal_torque = float(
        tau_ff_sequence[0]
    )


    if solve_ok:

        correction_torque = float(
            delta_tau.value[0]
        )


    else:

        solver_failures += 1

        # Safe fallback:
        # execute nominal inverse-dynamics feedforward only.
        #
        # IMPORTANT:
        # failures are still counted, so this cannot
        # "fake" successful MPC performance.

        correction_torque = 0.0


    torque_command = (

        nominal_torque
        + correction_torque
    )


    torque_command = float(

        np.clip(
            torque_command,
            -TORQUE_LIMIT,
            TORQUE_LIMIT
        )
    )


    # Final safety enforcement of torque-rate limit

    torque_command = float(

        np.clip(

            torque_command,

            previous_torque
            - MAX_TORQUE_STEP,

            previous_torque
            + MAX_TORQUE_STEP
        )
    )


    # --------------------------------------------------------
    # Apply first action
    # --------------------------------------------------------

    data.ctrl[
        ACTUATOR_ID
    ] = torque_command


    # Other joints retain original position servo

    data.ctrl[0] = 0.0
    data.ctrl[2:7] = 0.0


    for _ in range(
        physics_steps_per_control
    ):

        mujoco.mj_step(
            model,
            data
        )


    # --------------------------------------------------------
    # Record AFTER execution
    # --------------------------------------------------------

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

        actual_q
        - qd_now,

        nominal_torque,

        correction_torque,

        torque_command,

        inertia,

        dynamic_bias,

        solve_time,

        status_string
    ])


    previous_torque = (
        torque_command
    )


# ============================================================
# Analysis
# ============================================================

numeric = np.asarray(

    [
        row[:-1]
        for row in records
    ],

    dtype=np.float64
)


time_arr = numeric[:, 0]

qd_arr = numeric[:, 1]
q_arr = numeric[:, 3]

error_arr = numeric[:, 5]

nominal_arr = numeric[:, 6]
correction_arr = numeric[:, 7]
torque_arr = numeric[:, 8]

inertia_arr = numeric[:, 9]

solve_time_arr = numeric[:, 11]


# ============================================================
# Metrics
# ============================================================

rms_error = float(

    np.sqrt(
        np.mean(
            error_arr**2
        )
    )
)


mask_after_1s = (

    time_arr
    >= 1.0
)


rms_after_1s = float(

    np.sqrt(

        np.mean(

            error_arr[
                mask_after_1s
            ] ** 2
        )
    )
)


mean_abs_error = float(

    np.mean(
        np.abs(
            error_arr
        )
    )
)


max_abs_error = float(

    np.max(
        np.abs(
            error_arr
        )
    )
)


peak_torque = float(

    np.max(
        np.abs(
            torque_arr
        )
    )
)


control_effort = float(

    np.sum(
        torque_arr**2
    )
    * CONTROL_DT
)


max_torque_step_observed = float(

    np.max(

        np.abs(

            np.diff(
                torque_arr
            )
        )
    )
)


average_solver_ms = float(

    np.mean(
        solve_time_arr
    )
    * 1000.0
)


p95_solver_ms = float(

    np.percentile(
        solve_time_arr,
        95
    )
    * 1000.0
)


max_solver_ms = float(

    np.max(
        solve_time_arr
    )
    * 1000.0
)


deadline_miss_rate = float(

    np.mean(

        solve_time_arr
        > CONTROL_DT
    )

    * 100.0
)


# ============================================================
# CSV
# ============================================================

csv_path = os.path.join(

    OUTPUT_DIR,

    "mpc_v3_tracking.csv"
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

        "actual_position_rad",
        "actual_velocity_rad_s",

        "position_error_rad",

        "nominal_feedforward_torque_Nm",

        "mpc_correction_torque_Nm",

        "actual_command_torque_Nm",

        "effective_inertia",

        "dynamic_bias_Nm",

        "solver_time_s",

        "solver_status"
    ])


    writer.writerows(
        records
    )


# ============================================================
# Terminal summary
# ============================================================

print(
    "\n========== MPC V3 RESULT =========="
)


print(
    f"RMS error               : "
    f"{rms_error:.6f} rad"
)


print(
    f"RMS error after 1 s     : "
    f"{rms_after_1s:.6f} rad"
)


print(
    f"Mean abs error          : "
    f"{mean_abs_error:.6f} rad"
)


print(
    f"Max abs error           : "
    f"{max_abs_error:.6f} rad"
)


print(
    f"Peak torque             : "
    f"{peak_torque:.6f} N·m"
)


print(
    f"Control effort          : "
    f"{control_effort:.6f} N²·m²·s"
)


print(
    f"Maximum torque step     : "
    f"{max_torque_step_observed:.6f} N·m"
)


print(
    f"Average solver time     : "
    f"{average_solver_ms:.3f} ms"
)


print(
    f"95% solver time         : "
    f"{p95_solver_ms:.3f} ms"
)


print(
    f"Maximum solver time     : "
    f"{max_solver_ms:.3f} ms"
)


print(
    f"Control period          : "
    f"{CONTROL_DT * 1000:.1f} ms"
)


print(
    f"Deadline miss rate      : "
    f"{deadline_miss_rate:.2f}%"
)


print(
    f"Solver failures         : "
    f"{solver_failures}"
)


print(
    f"Solver status counts    : "
    f"{dict(status_counter)}"
)


# ============================================================
# Plot 1: tracking
# ============================================================

plt.figure(
    figsize=(9, 5)
)


plt.plot(
    time_arr,
    qd_arr,
    linestyle="--",
    label="Desired trajectory"
)


plt.plot(
    time_arr,
    q_arr,
    label="MPC v3"
)


plt.xlabel(
    "Time (s)"
)


plt.ylabel(
    "Joint 2 Position (rad)"
)


plt.title(
    "MPC V3: Error-State Receding-Horizon Tracking"
)


plt.legend()
plt.grid(True)
plt.tight_layout()


position_fig = os.path.join(

    OUTPUT_DIR,

    "mpc_v3_tracking_position.png"
)


plt.savefig(
    position_fig,
    dpi=200
)


# ============================================================
# Plot 2: error
# ============================================================

plt.figure(
    figsize=(9, 5)
)


plt.plot(
    time_arr,
    error_arr
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
    "MPC V3 Tracking Error"
)


plt.grid(True)
plt.tight_layout()


error_fig = os.path.join(

    OUTPUT_DIR,

    "mpc_v3_tracking_error.png"
)


plt.savefig(
    error_fig,
    dpi=200
)


# ============================================================
# Plot 3: torque decomposition
# ============================================================

plt.figure(
    figsize=(9, 5)
)


plt.plot(
    time_arr,
    nominal_arr,
    label="Nominal inverse-dynamics torque"
)


plt.plot(
    time_arr,
    correction_arr,
    label="MPC correction torque"
)


plt.plot(
    time_arr,
    torque_arr,
    label="Actual command torque"
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


plt.xlabel(
    "Time (s)"
)


plt.ylabel(
    "Torque (N·m)"
)


plt.title(
    "MPC V3: Nominal + Optimization Correction"
)


plt.legend()
plt.grid(True)
plt.tight_layout()


torque_fig = os.path.join(

    OUTPUT_DIR,

    "mpc_v3_torque_components.png"
)


plt.savefig(
    torque_fig,
    dpi=200
)


print(
    "\n[PASS] MPC v3 experiment finished"
)


print(
    "\nCSV saved to:"
)


print(
    csv_path
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
