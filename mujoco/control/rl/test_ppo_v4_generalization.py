import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from panda_joint2_env_v4 import PandaJoint2EnvV4


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "ppo_joint2_v4_100k.zip"
)
RESULT_DIR = PROJECT_ROOT / "outputs" / "rl" / "ppo_v4_generalization"

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# Load trained PPO v4
# ============================================================

model = PPO.load(
    MODEL_PATH,
    device="cpu"
)


# ============================================================
# Test scenarios
#
# NONE of these changed conditions are used for retraining.
# ============================================================

SCENARIOS = [

    {
        "name": "Nominal",
        "frequency": 0.50,
        "amplitude": 0.25,
        "center": -0.50,
        "initial_offset": 0.0,
        "initial_velocity_offset": 0.0,
        "external_torque": 0.0,
    },

    {
        "name": "Faster_0.65Hz",
        "frequency": 0.65,
        "amplitude": 0.25,
        "center": -0.50,
        "initial_offset": 0.0,
        "initial_velocity_offset": 0.0,
        "external_torque": 0.0,
    },

    {
        "name": "Slower_0.35Hz",
        "frequency": 0.35,
        "amplitude": 0.25,
        "center": -0.50,
        "initial_offset": 0.0,
        "initial_velocity_offset": 0.0,
        "external_torque": 0.0,
    },

    {
        "name": "Larger_amp_0.30",
        "frequency": 0.50,
        "amplitude": 0.30,
        "center": -0.50,
        "initial_offset": 0.0,
        "initial_velocity_offset": 0.0,
        "external_torque": 0.0,
    },

    {
        "name": "Shifted_center",
        "frequency": 0.50,
        "amplitude": 0.25,
        "center": -0.60,
        "initial_offset": 0.0,
        "initial_velocity_offset": 0.0,
        "external_torque": 0.0,
    },

    {
        "name": "Large_initial_error",
        "frequency": 0.50,
        "amplitude": 0.25,
        "center": -0.50,
        "initial_offset": 0.15,
        "initial_velocity_offset": -0.50,
        "external_torque": 0.0,
    },

    {
        "name": "External_5Nm",
        "frequency": 0.50,
        "amplitude": 0.25,
        "center": -0.50,
        "initial_offset": 0.0,
        "initial_velocity_offset": 0.0,
        "external_torque": 5.0,
    },
]


# ============================================================
# Configure scenario
# ============================================================

def configure_env(
    scenario,
    seed
):

    env = PandaJoint2EnvV4()

    # --------------------------------------------------------
    # Change reference trajectory
    # --------------------------------------------------------

    env.frequency = scenario[
        "frequency"
    ]

    env.amplitude = scenario[
        "amplitude"
    ]

    env.center = scenario[
        "center"
    ]

    env.omega = (
        2.0
        * np.pi
        * env.frequency
    )

    obs, info = env.reset(
        seed=seed
    )

    # --------------------------------------------------------
    # Additional initial-state disturbance
    # --------------------------------------------------------

    env.data.qpos[
        env.joint_id
    ] += scenario[
        "initial_offset"
    ]

    env.data.qvel[
        env.joint_id
    ] += scenario[
        "initial_velocity_offset"
    ]

    # --------------------------------------------------------
    # Constant generalized disturbance torque
    # --------------------------------------------------------

    env.data.qfrc_applied[
        env.joint_id
    ] = scenario[
        "external_torque"
    ]

    import mujoco

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

    episode_metrics = []

    representative = None


    for episode in range(
        episodes
    ):

        env, obs = configure_env(
            scenario,
            seed=2000 + episode
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

                "qd":
                    np.asarray(
                        qds
                    )
            }


        env.close()


    # --------------------------------------------------------
    # Mean result
    # --------------------------------------------------------

    def mean(key):

        return float(
            np.mean(
                [
                    metric[key]
                    for metric
                    in episode_metrics
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
                        metric[
                            "terminated"
                        ]
                        for metric
                        in episode_metrics
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
# Run suite
# ============================================================

print(
    "\n========== PPO V4 ZERO-SHOT GENERALIZATION =========="
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


    print(
        f"\n--- {summary['name']} ---"
    )

    print(
        f"Mean RMS error       : "
        f"{summary['rms']:.6f} rad"
    )

    print(
        f"Mean max error       : "
        f"{summary['max_error']:.6f} rad"
    )

    print(
        f"Mean peak torque     : "
        f"{summary['peak_torque']:.6f} N*m"
    )

    print(
        f"Mean control effort  : "
        f"{summary['control_effort']:.3f}"
    )

    print(
        f"Mean max torque step : "
        f"{summary['max_torque_step']:.6f} N*m"
    )

    print(
        f"Episode length       : "
        f"{summary['episode_length']:.2f} / 200"
    )

    print(
        f"Termination rate     : "
        f"{summary['termination_rate']:.1f}%"
    )


# ============================================================
# Plot 1:
# RMS error across scenarios
# ============================================================

names = [
    result[
        "name"
    ]
    for result in summaries
]

rms_values = [
    result[
        "rms"
    ]
    for result in summaries
]


plt.figure(
    figsize=(11, 5)
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
    "PPO V4 Zero-Shot Generalization"
)

plt.grid(
    True,
    axis="y"
)

plt.tight_layout()


rms_fig = os.path.join(
    RESULT_DIR,
    "generalization_rms.png"
)

plt.savefig(
    rms_fig,
    dpi=200
)


# ============================================================
# Plot 2:
# Representative tracking traces
# ============================================================

plt.figure(
    figsize=(10, 6)
)


for scenario_name in [
    "Nominal",
    "Faster_0.65Hz",
    "Large_initial_error",
    "External_5Nm",
]:

    trace = traces[
        scenario_name
    ]

    plt.plot(
        trace[
            "time"
        ],

        trace[
            "q"
        ],

        label=scenario_name
    )


plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Joint 2 Position (rad)"
)

plt.title(
    "PPO V4: Representative Unseen Conditions"
)

plt.legend()
plt.grid(True)
plt.tight_layout()


trace_fig = os.path.join(
    RESULT_DIR,
    "generalization_traces.png"
)

plt.savefig(
    trace_fig,
    dpi=200
)


print(
    "\n[PASS] Generalization test finished"
)

print(
    "\nFigures saved to:"
)

print(
    rms_fig
)

print(
    trace_fig
)
