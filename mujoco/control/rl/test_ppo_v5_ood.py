import os
from pathlib import Path

import mujoco
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from panda_joint2_env_v5 import PandaJoint2EnvV5


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "ppo_joint2_v5_150k.zip"
)
RESULT_DIR = PROJECT_ROOT / "outputs" / "rl" / "ppo_v5_ood"

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# Load trained PPO V5
# ============================================================

model = PPO.load(
    MODEL_PATH,
    device="cpu"
)


# ============================================================
# OOD test scenarios
#
# PPO V5 training distribution:
#
# frequency       0.35 ~ 0.65 Hz
# amplitude       0.20 ~ 0.30 rad
# center         -0.60 ~ -0.40 rad
# q offset       -0.15 ~ +0.15 rad
# dq offset      -0.50 ~ +0.50 rad/s
# external torque -5 ~ +5 N*m
# damping scale   0.80 ~ 1.20
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
        "name": "OOD_Faster_0.75Hz",

        "frequency": 0.75,
        "amplitude": 0.25,
        "center": -0.50,

        "q_offset": 0.00,
        "dq_offset": 0.00,

        "external_torque": 0.0,
        "damping_scale": 1.00,
    },

    {
        "name": "OOD_Larger_amp_0.34",

        "frequency": 0.50,
        "amplitude": 0.34,
        "center": -0.50,

        "q_offset": 0.00,
        "dq_offset": 0.00,

        "external_torque": 0.0,
        "damping_scale": 1.00,
    },

    {
        "name": "OOD_Center_-0.68",

        "frequency": 0.50,
        "amplitude": 0.25,
        "center": -0.68,

        "q_offset": 0.00,
        "dq_offset": 0.00,

        "external_torque": 0.0,
        "damping_scale": 1.00,
    },

    {
        "name": "OOD_Initial_error",

        "frequency": 0.50,
        "amplitude": 0.25,
        "center": -0.50,

        "q_offset": 0.20,
        "dq_offset": -0.70,

        "external_torque": 0.0,
        "damping_scale": 1.00,
    },

    {
        "name": "OOD_External_8Nm",

        "frequency": 0.50,
        "amplitude": 0.25,
        "center": -0.50,

        "q_offset": 0.00,
        "dq_offset": 0.00,

        "external_torque": 8.0,
        "damping_scale": 1.00,
    },

    {
        "name": "OOD_Damping_1.30",

        "frequency": 0.50,
        "amplitude": 0.25,
        "center": -0.50,

        "q_offset": 0.00,
        "dq_offset": 0.00,

        "external_torque": 0.0,
        "damping_scale": 1.30,
    },

    {
        "name": "OOD_Combined",

        "frequency": 0.75,
        "amplitude": 0.34,
        "center": -0.66,

        "q_offset": 0.20,
        "dq_offset": -0.70,

        "external_torque": 8.0,
        "damping_scale": 1.30,
    },
]


# ============================================================
# Configure exact scenario
# ============================================================

def configure_env(
    scenario,
    seed
):

    env = PandaJoint2EnvV5()

    env.reset(
        seed=seed
    )

    dof_id = int(
        env.model.jnt_dofadr[
            env.joint_id
        ]
    )

    # --------------------------------------------------------
    # Clear MuJoCo dynamic state
    # --------------------------------------------------------

    mujoco.mj_resetData(
        env.model,
        env.data
    )

    # --------------------------------------------------------
    # Task
    # --------------------------------------------------------

    env.frequency = float(
        scenario["frequency"]
    )

    env.amplitude = float(
        scenario["amplitude"]
    )

    env.center = float(
        scenario["center"]
    )

    env.omega = (
        2.0
        * np.pi
        * env.frequency
    )

    # --------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------

    env.damping_scale = float(
        scenario["damping_scale"]
    )

    env.model.dof_damping[
        dof_id
    ] = (
        env.nominal_damping
        * env.damping_scale
    )

    # --------------------------------------------------------
    # Tiny common noise between evaluation episodes
    # --------------------------------------------------------

    rng = np.random.default_rng(
        seed
    )

    q_noise = rng.uniform(
        -0.01,
        0.01
    )

    dq_noise = rng.uniform(
        -0.03,
        0.03
    )

    (
        q_des_0,
        dq_des_0,
        _
    ) = env.desired_trajectory(
        0.0
    )

    env.data.qpos[:] = 0.0
    env.data.qvel[:] = 0.0
    env.data.ctrl[:] = 0.0
    env.data.qfrc_applied[:] = 0.0

    env.data.qpos[
        env.joint_id
    ] = (
        q_des_0
        + scenario["q_offset"]
        + q_noise
    )

    env.data.qvel[
        dof_id
    ] = (
        dq_des_0
        + scenario["dq_offset"]
        + dq_noise
    )

    # --------------------------------------------------------
    # External disturbance
    # --------------------------------------------------------

    env.external_torque = float(
        scenario[
            "external_torque"
        ]
    )

    env.data.qfrc_applied[
        dof_id
    ] = env.external_torque

    # --------------------------------------------------------
    # Controller state
    # --------------------------------------------------------

    env.current_torque = 0.0

    env.data.ctrl[
        env.actuator_id
    ] = 0.0

    env.step_count = 0

    mujoco.mj_forward(
        env.model,
        env.data
    )

    obs = env._get_observation()

    return (
        env,
        obs
    )


# ============================================================
# Evaluate one scenario
# ============================================================

def evaluate_scenario(
    scenario,
    episodes=10
):

    metrics = []

    representative = None


    for episode in range(
        episodes
    ):

        env, obs = configure_env(
            scenario,
            seed=5000 + episode
        )

        errors = []
        torques = []
        torque_steps = []

        times = []
        qs = []
        qds = []

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
                info[
                    "position_error"
                ]
            )

            torques.append(
                info[
                    "torque"
                ]
            )

            torque_steps.append(
                info[
                    "torque_change"
                ]
            )

            times.append(
                info[
                    "time"
                ]
            )

            qs.append(
                info[
                    "q"
                ]
            )

            qds.append(
                info[
                    "q_des"
                ]
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


        metrics.append({

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
                        qs
                    ),

                "q_des":
                    np.asarray(
                        qds
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
                    m[key]
                    for m in metrics
                ]
            )
        )


    summary = {

        "name":
            scenario[
                "name"
            ],

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

        "episode_length":
            mean(
                "length"
            ),

        "termination_rate":
            float(
                np.mean(
                    [
                        m[
                            "terminated"
                        ]
                        for m in metrics
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
# Run all scenarios
# ============================================================

print(
    "\n========== PPO V5 OOD STRESS TEST =========="
)

summaries = []

traces = {}


for scenario in SCENARIOS:

    summary, trace = evaluate_scenario(
        scenario,
        episodes=10
    )

    summaries.append(
        summary
    )

    traces[
        scenario[
            "name"
        ]
    ] = trace


# ============================================================
# Nominal RMS baseline
# ============================================================

nominal_rms = summaries[0][
    "rms"
]


# ============================================================
# Print results
# ============================================================

for summary in summaries:

    degradation_ratio = (
        summary[
            "rms"
        ]
        / nominal_rms
    )

    print(
        f"\n--- {summary['name']} ---"
    )

    print(
        f"Mean RMS error        : "
        f"{summary['rms']:.6f} rad"
    )

    print(
        f"Mean max error        : "
        f"{summary['max_error']:.6f} rad"
    )

    print(
        f"Peak torque           : "
        f"{summary['peak_torque']:.3f} N*m"
    )

    print(
        f"Control effort        : "
        f"{summary['control_effort']:.3f}"
    )

    print(
        f"Max torque step       : "
        f"{summary['max_torque_step']:.6f} N*m"
    )

    print(
        f"Episode length        : "
        f"{summary['episode_length']:.2f} / 200"
    )

    print(
        f"Termination rate      : "
        f"{summary['termination_rate']:.1f}%"
    )

    print(
        f"Degradation ratio     : "
        f"{degradation_ratio:.2f} x nominal"
    )


# ============================================================
# Overall OOD summary
#
# Exclude nominal.
# ============================================================

ood_summaries = summaries[
    1:
]


average_ood_rms = float(
    np.mean(
        [
            s[
                "rms"
            ]
            for s
            in ood_summaries
        ]
    )
)

worst_ood_rms = float(
    np.max(
        [
            s[
                "rms"
            ]
            for s
            in ood_summaries
        ]
    )
)

worst_scenario = max(
    ood_summaries,
    key=lambda x: x[
        "rms"
    ]
)


print(
    "\n========== OOD SUMMARY =========="
)

print(
    f"Nominal RMS              : "
    f"{nominal_rms:.6f} rad"
)

print(
    f"Average OOD RMS          : "
    f"{average_ood_rms:.6f} rad"
)

print(
    f"Worst OOD RMS            : "
    f"{worst_ood_rms:.6f} rad"
)

print(
    f"Worst OOD scenario       : "
    f"{worst_scenario['name']}"
)

print(
    f"Worst degradation ratio  : "
    f"{worst_ood_rms / nominal_rms:.2f} x"
)


# ============================================================
# Plot RMS
# ============================================================

names = [
    s[
        "name"
    ]
    for s
    in summaries
]

rms_values = [
    s[
        "rms"
    ]
    for s
    in summaries
]


plt.figure(
    figsize=(12, 6)
)

plt.bar(
    names,
    rms_values
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.ylabel(
    "Mean RMS Error (rad)"
)

plt.title(
    "PPO V5 Out-of-Distribution Stress Test"
)

plt.grid(
    True,
    axis="y"
)

plt.tight_layout()


rms_fig = os.path.join(
    RESULT_DIR,
    "ppo_v5_ood_rms.png"
)

plt.savefig(
    rms_fig,
    dpi=200
)


# ============================================================
# OOD Combined tracking
# ============================================================

trace = traces[
    "OOD_Combined"
]


plt.figure(
    figsize=(9, 5)
)

plt.plot(
    trace[
        "time"
    ],

    trace[
        "q_des"
    ],

    linestyle="--",
    label="Desired trajectory"
)

plt.plot(
    trace[
        "time"
    ],

    trace[
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
    "PPO V5: OOD Combined Challenge"
)

plt.legend()
plt.grid(True)
plt.tight_layout()


tracking_fig = os.path.join(
    RESULT_DIR,
    "ppo_v5_ood_combined_tracking.png"
)

plt.savefig(
    tracking_fig,
    dpi=200
)


# ============================================================
# OOD Combined torque
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    trace[
        "time"
    ],

    trace[
        "torque"
    ],

    label="PPO v5 torque"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Torque (N*m)"
)

plt.title(
    "PPO V5: OOD Combined Torque"
)

plt.legend()
plt.grid(True)
plt.tight_layout()


torque_fig = os.path.join(
    RESULT_DIR,
    "ppo_v5_ood_combined_torque.png"
)

plt.savefig(
    torque_fig,
    dpi=200
)


print(
    "\n[PASS] PPO v5 OOD stress test finished"
)

print(
    "\nFigures saved to:"
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
