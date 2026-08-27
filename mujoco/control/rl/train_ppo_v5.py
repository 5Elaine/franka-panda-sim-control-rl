import os
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mujoco

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from panda_joint2_env_v4 import PandaJoint2EnvV4
from panda_joint2_env_v5 import PandaJoint2EnvV5


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = PROJECT_ROOT / "outputs" / "rl" / "ppo_v5"

LOG_DIR = os.path.join(
    BASE_DIR,
    "ppo_v5_logs"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "ppo_v5_models"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "ppo_v5_results"
)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


MONITOR_FILE = os.path.join(
    LOG_DIR,
    "train.monitor.csv"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ppo_joint2_v5_150k"
)

V4_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "ppo_joint2_v4_100k.zip"
)


# ============================================================
# Training environment
# ============================================================

train_env = Monitor(
    PandaJoint2EnvV5(),
    filename=MONITOR_FILE
)


# ============================================================
# PPO
#
# Keep the same basic network / PPO settings as v4.
#
# This is important:
# if we changed everything at once, we would not know whether
# improvement came from domain randomization or other changes.
# ============================================================

model_v5 = PPO(

    policy="MlpPolicy",

    env=train_env,

    learning_rate=3e-4,

    n_steps=2048,

    batch_size=256,

    n_epochs=10,

    gamma=0.99,

    gae_lambda=0.95,

    clip_range=0.2,

    ent_coef=0.0,

    vf_coef=0.5,

    max_grad_norm=0.5,

    policy_kwargs=dict(

        net_arch=dict(

            pi=[
                64,
                64
            ],

            vf=[
                64,
                64
            ]
        )
    ),

    verbose=1,

    device="cpu",

    seed=42
)


print(
    "\n========== PPO V5 DOMAIN-RANDOMIZED TRAINING =========="
)

print(
    "Training timesteps       : 150000"
)

print(
    "Observation dimension    : 11"
)

print(
    "Action                   : delta torque"
)

print(
    "Torque limit             : ±60 N*m"
)

print(
    "Torque-step limit        : ±10 N*m / 20 ms"
)

print(
    "\nTask randomization:"
)

print(
    "  Frequency              : 0.35 ~ 0.65 Hz"
)

print(
    "  Amplitude              : 0.20 ~ 0.30 rad"
)

print(
    "  Center                 : -0.60 ~ -0.40 rad"
)

print(
    "\nInitial-state randomization:"
)

print(
    "  Position offset        : ±0.15 rad"
)

print(
    "  Velocity offset        : ±0.50 rad/s"
)

print(
    "\nDomain randomization:"
)

print(
    "  External torque        : ±5 N*m"
)

print(
    "  Joint damping scale    : 0.80 ~ 1.20"
)

print(
    "\nTraining starts...\n"
)


# ============================================================
# TRAIN
# ============================================================

model_v5.learn(
    total_timesteps=150_000,
    progress_bar=True
)


model_v5.save(
    MODEL_PATH
)

train_env.close()


print(
    "\n[PASS] PPO v5 training finished"
)

print(
    f"Model saved to:\n{MODEL_PATH}.zip"
)


# ============================================================
# Load PPO v4
# ============================================================

model_v4 = PPO.load(
    V4_MODEL_PATH,
    device="cpu"
)


# ============================================================
# Common benchmark scenarios
#
# v4:
#   These are mostly zero-shot conditions.
#
# v5:
#   Most individual conditions are inside its randomized
#   training distribution.
#
# Combined challenge:
#   tests simultaneous variation of many factors.
# ============================================================

SCENARIOS = [

    {
        "name": "Nominal",

        "frequency": 0.50,
        "amplitude": 0.25,
        "center": -0.50,

        "q_offset": 0.00,
        "dq_offset": 0.00,

        "external_torque": 0.0,

        "damping_scale": 1.00,
    },

    {
        "name": "Faster_0.65Hz",

        "frequency": 0.65,
        "amplitude": 0.25,
        "center": -0.50,

        "q_offset": 0.00,
        "dq_offset": 0.00,

        "external_torque": 0.0,

        "damping_scale": 1.00,
    },

    {
        "name": "Slower_0.35Hz",

        "frequency": 0.35,
        "amplitude": 0.25,
        "center": -0.50,

        "q_offset": 0.00,
        "dq_offset": 0.00,

        "external_torque": 0.0,

        "damping_scale": 1.00,
    },

    {
        "name": "Larger_amp_0.30",

        "frequency": 0.50,
        "amplitude": 0.30,
        "center": -0.50,

        "q_offset": 0.00,
        "dq_offset": 0.00,

        "external_torque": 0.0,

        "damping_scale": 1.00,
    },

    {
        "name": "Shifted_center",

        "frequency": 0.50,
        "amplitude": 0.25,
        "center": -0.60,

        "q_offset": 0.00,
        "dq_offset": 0.00,

        "external_torque": 0.0,

        "damping_scale": 1.00,
    },

    {
        "name": "Large_initial_error",

        "frequency": 0.50,
        "amplitude": 0.25,
        "center": -0.50,

        "q_offset": 0.15,
        "dq_offset": -0.50,

        "external_torque": 0.0,

        "damping_scale": 1.00,
    },

    {
        "name": "External_5Nm",

        "frequency": 0.50,
        "amplitude": 0.25,
        "center": -0.50,

        "q_offset": 0.00,
        "dq_offset": 0.00,

        "external_torque": 5.0,

        "damping_scale": 1.00,
    },

    {
        "name": "Damping_1.20",

        "frequency": 0.50,
        "amplitude": 0.25,
        "center": -0.50,

        "q_offset": 0.00,
        "dq_offset": 0.00,

        "external_torque": 0.0,

        "damping_scale": 1.20,
    },

    {
        "name": "Combined_challenge",

        "frequency": 0.62,
        "amplitude": 0.29,
        "center": -0.58,

        "q_offset": 0.12,
        "dq_offset": -0.40,

        "external_torque": 4.5,

        "damping_scale": 1.18,
    },
]


# ============================================================
# Configure V4 / V5 into EXACTLY the same benchmark scenario
#
# We deliberately override each environment after reset.
# ============================================================

def configure_scenario(
    env,
    scenario,
    seed
):

    # --------------------------------------------------------
    # Call reset once so Gymnasium / RNG / MuJoCo bookkeeping
    # is initialized correctly.
    # --------------------------------------------------------

    env.reset(
        seed=seed
    )

    # --------------------------------------------------------
    # Locate correct DOF
    # --------------------------------------------------------

    dof_id = int(
        env.model.jnt_dofadr[
            env.joint_id
        ]
    )

    # --------------------------------------------------------
    # Save / obtain nominal damping
    # --------------------------------------------------------

    if hasattr(
        env,
        "nominal_damping"
    ):

        nominal_damping = float(
            env.nominal_damping
        )

    else:

        # Panda nominal damping for this model is restored
        # from a fresh model instance.
        fresh_model = mujoco.MjModel.from_xml_path(
            env.model_path
        )

        nominal_damping = float(
            fresh_model.dof_damping[
                dof_id
            ]
        )

    # --------------------------------------------------------
    # Reset dynamic state again before applying benchmark
    # --------------------------------------------------------

    mujoco.mj_resetData(
        env.model,
        env.data
    )

    # --------------------------------------------------------
    # Exact benchmark task
    # --------------------------------------------------------

    env.frequency = float(
        scenario[
            "frequency"
        ]
    )

    env.amplitude = float(
        scenario[
            "amplitude"
        ]
    )

    env.center = float(
        scenario[
            "center"
        ]
    )

    env.omega = (
        2.0
        * np.pi
        * env.frequency
    )

    # --------------------------------------------------------
    # Exact benchmark dynamics
    # --------------------------------------------------------

    env.model.dof_damping[
        dof_id
    ] = (
        nominal_damping
        * scenario[
            "damping_scale"
        ]
    )

    # --------------------------------------------------------
    # Exact benchmark initial state
    #
    # Add only a tiny common perturbation so that the 10
    # episodes are not mathematically identical.
    # BOTH policies receive the same perturbation.
    # --------------------------------------------------------

    rng = np.random.default_rng(
        seed
    )

    small_q_noise = rng.uniform(
        -0.01,
        0.01
    )

    small_dq_noise = rng.uniform(
        -0.03,
        0.03
    )

    desired = env.desired_trajectory(
        0.0
    )

    q_des_0 = desired[0]
    dq_des_0 = desired[1]

    env.data.qpos[:] = 0.0
    env.data.qvel[:] = 0.0
    env.data.ctrl[:] = 0.0
    env.data.qfrc_applied[:] = 0.0

    env.data.qpos[
        env.joint_id
    ] = (
        q_des_0
        + scenario[
            "q_offset"
        ]
        + small_q_noise
    )

    env.data.qvel[
        dof_id
    ] = (
        dq_des_0
        + scenario[
            "dq_offset"
        ]
        + small_dq_noise
    )

    # --------------------------------------------------------
    # Control state
    # --------------------------------------------------------

    env.current_torque = 0.0

    env.data.ctrl[
        env.actuator_id
    ] = 0.0

    env.step_count = 0

    # --------------------------------------------------------
    # External disturbance
    #
    # V5 step() explicitly reapplies self.external_torque,
    # so set that attribute too.
    # --------------------------------------------------------

    external_torque = float(
        scenario[
            "external_torque"
        ]
    )

    env.data.qfrc_applied[
        dof_id
    ] = external_torque

    if hasattr(
        env,
        "external_torque"
    ):

        env.external_torque = (
            external_torque
        )

    # --------------------------------------------------------
    # Keep V5 diagnostic attributes consistent
    # --------------------------------------------------------

    if hasattr(
        env,
        "damping_scale"
    ):

        env.damping_scale = float(
            scenario[
                "damping_scale"
            ]
        )

    # --------------------------------------------------------
    # Forward dynamics
    # --------------------------------------------------------

    mujoco.mj_forward(
        env.model,
        env.data
    )

    obs = env._get_observation()

    return (
        obs,
        dof_id
    )


# ============================================================
# Evaluate a model on one scenario
# ============================================================

def evaluate_scenario(
    model,
    env_class,
    scenario,
    episodes=10
):

    episode_metrics = []

    representative = None


    for episode in range(
        episodes
    ):

        env = env_class()

        seed = (
            3000
            + episode
        )

        obs, dof_id = configure_scenario(
            env,
            scenario,
            seed
        )

        errors = []
        torques = []
        torque_steps = []

        times = []
        q_values = []
        q_des_values = []

        terminated = False
        truncated = False


        while not (
            terminated
            or truncated
        ):

            action, _ = model.predict(
                obs,
                deterministic=True
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(
                action
            )

            errors.append(
                float(
                    info[
                        "position_error"
                    ]
                )
            )

            torques.append(
                float(
                    info[
                        "torque"
                    ]
                )
            )

            torque_steps.append(
                float(
                    info[
                        "torque_change"
                    ]
                )
            )

            times.append(
                float(
                    info[
                        "time"
                    ]
                )
            )

            q_values.append(
                float(
                    info[
                        "q"
                    ]
                )
            )

            q_des_values.append(
                float(
                    info[
                        "q_des"
                    ]
                )
            )


        errors = np.asarray(
            errors
        )

        torques = np.asarray(
            torques
        )

        torque_steps = np.asarray(
            torque_steps
        )


        episode_metrics.append({

            "rms":
                float(
                    np.sqrt(
                        np.mean(
                            errors ** 2
                        )
                    )
                ),

            "max_error":
                float(
                    np.max(
                        np.abs(
                            errors
                        )
                    )
                ),

            "peak_torque":
                float(
                    np.max(
                        np.abs(
                            torques
                        )
                    )
                ),

            "control_effort":
                float(
                    np.sum(
                        torques ** 2
                    )
                    * env.control_dt
                ),

            "max_torque_step":
                float(
                    np.max(
                        np.abs(
                            torque_steps
                        )
                    )
                ),

            "mean_abs_torque_step":
                float(
                    np.mean(
                        np.abs(
                            torque_steps
                        )
                    )
                ),

            "length":
                len(
                    errors
                ),

            "terminated":
                bool(
                    terminated
                )
        })


        if episode == 0:

            representative = {

                "time":
                    np.asarray(
                        times
                    ),

                "q":
                    np.asarray(
                        q_values
                    ),

                "q_des":
                    np.asarray(
                        q_des_values
                    ),

                "torque":
                    np.asarray(
                        torques
                    )
            }


        env.close()


    def mean(
        key
    ):

        return float(
            np.mean(
                [
                    item[key]
                    for item in episode_metrics
                ]
            )
        )


    summary = {

        "rms":
            mean(
                "rms"
            ),

        "max_error":
            mean(
                "max_error"
            ),

        "peak_torque":
            mean(
                "peak_torque"
            ),

        "control_effort":
            mean(
                "control_effort"
            ),

        "max_torque_step":
            mean(
                "max_torque_step"
            ),

        "mean_abs_torque_step":
            mean(
                "mean_abs_torque_step"
            ),

        "episode_length":
            mean(
                "length"
            ),

        "termination_rate":
            float(
                np.mean(
                    [
                        item[
                            "terminated"
                        ]
                        for item in episode_metrics
                    ]
                )
                * 100.0
            )
    }


    return (
        summary,
        representative
    )


# ============================================================
# Run common benchmark
# ============================================================

print(
    "\n========== V4 vs V5 COMMON ROBUSTNESS BENCHMARK =========="
)


benchmark_results = []

combined_traces = {}


for scenario in SCENARIOS:

    v4_summary, v4_trace = (
        evaluate_scenario(
            model_v4,
            PandaJoint2EnvV4,
            scenario
        )
    )

    v5_summary, v5_trace = (
        evaluate_scenario(
            model_v5,
            PandaJoint2EnvV5,
            scenario
        )
    )

    benchmark_results.append({

        "name":
            scenario[
                "name"
            ],

        "v4":
            v4_summary,

        "v5":
            v5_summary
    })


    if scenario[
        "name"
    ] == "Combined_challenge":

        combined_traces[
            "v4"
        ] = v4_trace

        combined_traces[
            "v5"
        ] = v5_trace


    print(
        f"\n--- {scenario['name']} ---"
    )

    print(
        "PPO v4:"
    )

    print(
        f"  RMS error             : "
        f"{v4_summary['rms']:.6f} rad"
    )

    print(
        f"  Max error             : "
        f"{v4_summary['max_error']:.6f} rad"
    )

    print(
        f"  Peak torque           : "
        f"{v4_summary['peak_torque']:.3f} N*m"
    )

    print(
        f"  Control effort        : "
        f"{v4_summary['control_effort']:.3f}"
    )

    print(
        f"  Max torque step       : "
        f"{v4_summary['max_torque_step']:.6f} N*m"
    )

    print(
        f"  Termination rate      : "
        f"{v4_summary['termination_rate']:.1f}%"
    )


    print(
        "PPO v5:"
    )

    print(
        f"  RMS error             : "
        f"{v5_summary['rms']:.6f} rad"
    )

    print(
        f"  Max error             : "
        f"{v5_summary['max_error']:.6f} rad"
    )

    print(
        f"  Peak torque           : "
        f"{v5_summary['peak_torque']:.3f} N*m"
    )

    print(
        f"  Control effort        : "
        f"{v5_summary['control_effort']:.3f}"
    )

    print(
        f"  Max torque step       : "
        f"{v5_summary['max_torque_step']:.6f} N*m"
    )

    print(
        f"  Termination rate      : "
        f"{v5_summary['termination_rate']:.1f}%"
    )


# ============================================================
# Aggregate robustness score
#
# Do NOT include only nominal.
# Average across ALL test scenarios.
# ============================================================

v4_average_rms = float(
    np.mean(
        [
            result[
                "v4"
            ][
                "rms"
            ]
            for result
            in benchmark_results
        ]
    )
)

v5_average_rms = float(
    np.mean(
        [
            result[
                "v5"
            ][
                "rms"
            ]
            for result
            in benchmark_results
        ]
    )
)

v4_worst_rms = float(
    np.max(
        [
            result[
                "v4"
            ][
                "rms"
            ]
            for result
            in benchmark_results
        ]
    )
)

v5_worst_rms = float(
    np.max(
        [
            result[
                "v5"
            ][
                "rms"
            ]
            for result
            in benchmark_results
        ]
    )
)


print(
    "\n========== ROBUSTNESS SUMMARY =========="
)

print(
    f"V4 average RMS across scenarios : "
    f"{v4_average_rms:.6f} rad"
)

print(
    f"V5 average RMS across scenarios : "
    f"{v5_average_rms:.6f} rad"
)

print(
    f"V4 worst-case RMS               : "
    f"{v4_worst_rms:.6f} rad"
)

print(
    f"V5 worst-case RMS               : "
    f"{v5_worst_rms:.6f} rad"
)


# ============================================================
# Learning curve
# ============================================================

episode_rewards = []


with open(
    MONITOR_FILE,
    "r"
) as f:

    reader = csv.DictReader(

        line
        for line in f
        if not line.startswith(
            "#"
        )
    )

    for row in reader:

        episode_rewards.append(
            float(
                row[
                    "r"
                ]
            )
        )


episode_rewards = np.asarray(
    episode_rewards
)


WINDOW = 20

kernel = (
    np.ones(
        WINDOW
    )
    / WINDOW
)

reward_ma = np.convolve(
    episode_rewards,
    kernel,
    mode="valid"
)

moving_x = np.arange(
    WINDOW - 1,
    len(
        episode_rewards
    )
)


plt.figure(
    figsize=(9, 5)
)

plt.plot(
    episode_rewards,
    alpha=0.25,
    label="Episode reward"
)

plt.plot(
    moving_x,
    reward_ma,
    linewidth=2,
    label="20-episode mean"
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Episode Reward"
)

plt.title(
    "PPO V5 Domain-Randomized Learning Curve"
)

plt.legend()
plt.grid(True)
plt.tight_layout()


learning_fig = os.path.join(
    RESULT_DIR,
    "ppo_v5_learning_curve.png"
)

plt.savefig(
    learning_fig,
    dpi=200
)


# ============================================================
# RMS comparison bar chart
# ============================================================

scenario_names = [
    result[
        "name"
    ]
    for result
    in benchmark_results
]

v4_rms = np.asarray(
    [
        result[
            "v4"
        ][
            "rms"
        ]
        for result
        in benchmark_results
    ]
)

v5_rms = np.asarray(
    [
        result[
            "v5"
        ][
            "rms"
        ]
        for result
        in benchmark_results
    ]
)


x = np.arange(
    len(
        scenario_names
    )
)

width = 0.38


plt.figure(
    figsize=(13, 6)
)

plt.bar(
    x - width / 2,
    v4_rms,
    width,
    label="PPO v4"
)

plt.bar(
    x + width / 2,
    v5_rms,
    width,
    label="PPO v5"
)

plt.xticks(
    x,
    scenario_names,
    rotation=30,
    ha="right"
)

plt.ylabel(
    "Mean RMS Error (rad)"
)

plt.title(
    "Fixed-Domain PPO V4 vs Randomized PPO V5"
)

plt.legend()
plt.grid(
    True,
    axis="y"
)

plt.tight_layout()


rms_fig = os.path.join(
    RESULT_DIR,
    "v4_vs_v5_rms.png"
)

plt.savefig(
    rms_fig,
    dpi=200
)


# ============================================================
# Combined challenge tracking
# ============================================================

v4_trace = combined_traces[
    "v4"
]

v5_trace = combined_traces[
    "v5"
]


plt.figure(
    figsize=(9, 5)
)

plt.plot(
    v5_trace[
        "time"
    ],

    v5_trace[
        "q_des"
    ],

    linestyle="--",
    label="Desired trajectory"
)

plt.plot(
    v4_trace[
        "time"
    ],

    v4_trace[
        "q"
    ],

    label="PPO v4"
)

plt.plot(
    v5_trace[
        "time"
    ],

    v5_trace[
        "q"
    ],

    label="PPO v5"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Joint 2 Position (rad)"
)

plt.title(
    "Combined Challenge: Tracking"
)

plt.legend()
plt.grid(True)
plt.tight_layout()


tracking_fig = os.path.join(
    RESULT_DIR,
    "v4_vs_v5_combined_tracking.png"
)

plt.savefig(
    tracking_fig,
    dpi=200
)


# ============================================================
# Combined challenge torque
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    v4_trace[
        "time"
    ],

    v4_trace[
        "torque"
    ],

    label="PPO v4"
)

plt.plot(
    v5_trace[
        "time"
    ],

    v5_trace[
        "torque"
    ],

    label="PPO v5"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Torque (N*m)"
)

plt.title(
    "Combined Challenge: Torque Command"
)

plt.legend()
plt.grid(True)
plt.tight_layout()


torque_fig = os.path.join(
    RESULT_DIR,
    "v4_vs_v5_combined_torque.png"
)

plt.savefig(
    torque_fig,
    dpi=200
)


print(
    "\n[PASS] PPO v5 training and benchmark finished"
)

print(
    "\nFigures saved to:"
)

print(
    learning_fig
)

print(
    rms_fig
)

print(
    tracking_fig
)

print(
    torque_fig
)
