import os
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from stable_baselines3.common.monitor import (
    Monitor
)

from stable_baselines3.common.env_checker import (
    check_env
)

from panda_joint2_env_v3 import (
    PandaJoint2EnvV3
)

from panda_joint2_env_v4 import (
    PandaJoint2EnvV4
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASE_DIR = PROJECT_ROOT / "outputs" / "rl" / "historical_v3_v4"

LOG_DIR = os.path.join(
    BASE_DIR,
    "ppo_v4_logs"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "ppo_v4_models"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "ppo_v4_results"
)

os.makedirs(
    LOG_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


MONITOR_FILE = os.path.join(
    LOG_DIR,
    "train.monitor.csv"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ppo_joint2_v4_100k"
)

V3_MODEL_PATH = os.path.join(
    BASE_DIR,
    "ppo_v3_models",
    "ppo_joint2_v3_100k.zip"
)


# ============================================================
# Environment validation
# ============================================================

test_env = PandaJoint2EnvV4()

check_env(
    test_env,
    warn=True
)

test_env.close()


print(
    "[PASS] PPO v4 environment check passed"
)


# ============================================================
# Training environment
# ============================================================

train_env = Monitor(
    PandaJoint2EnvV4(),
    filename=MONITOR_FILE
)


model_v4 = PPO(

    "MlpPolicy",

    train_env,

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
    "\n========== PPO V4 TRAINING =========="
)

print(
    "Timesteps             : 100000"
)

print(
    "Action meaning        : torque increment"
)

print(
    "Absolute torque limit : ±60 N*m"
)

print(
    "Hard torque step limit: ±10 N*m / 20 ms"
)

print(
    "Observation           : state + current torque + phase + time"
)


model_v4.learn(
    total_timesteps=100_000,
    progress_bar=True
)


model_v4.save(
    MODEL_PATH
)

train_env.close()


print(
    "\n[PASS] PPO v4 training finished"
)


# ============================================================
# Load PPO v3
# ============================================================

model_v3 = PPO.load(
    V3_MODEL_PATH,
    device="cpu"
)


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    env_class,
    episodes=10
):

    env = env_class()

    results = []

    representative = None


    for episode in range(
        episodes
    ):

        obs, _ = env.reset(
            seed=1000 + episode
        )

        errors = []
        torques = []
        torque_steps = []
        rewards = []

        times = []
        qs = []
        qds = []

        absolute_limit_active = []

        terminated = False
        truncated = False

        previous_torque = 0.0


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

            torque = float(
                info[
                    "torque"
                ]
            )

            torque_step = (
                torque
                - previous_torque
            )

            errors.append(
                info[
                    "position_error"
                ]
            )

            torques.append(
                torque
            )

            torque_steps.append(
                torque_step
            )

            rewards.append(
                reward
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

            absolute_limit_active.append(
                bool(
                    info.get(
                        "absolute_limit_active",
                        False
                    )
                )
            )

            previous_torque = (
                torque
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


        results.append({

            "rms":
                float(
                    np.sqrt(
                        np.mean(
                            errors ** 2
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

            "episode_length":
                len(
                    rewards
                ),

            "terminated":
                bool(
                    terminated
                ),

            "absolute_limit_fraction":
                float(
                    np.mean(
                        absolute_limit_active
                    )
                    * 100.0
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
                    ),

                "torque":
                    torques,

                "torque_step":
                    torque_steps
            }


    env.close()

    return (
        results,
        representative
    )


# ============================================================
# Evaluate v3 / v4
# ============================================================

v3_results, v3_trace = evaluate(
    model_v3,
    PandaJoint2EnvV3
)

v4_results, v4_trace = evaluate(
    model_v4,
    PandaJoint2EnvV4
)


# ============================================================
# Summary
# ============================================================

def summarize(
    name,
    results
):

    def mean(key):

        return float(
            np.mean(
                [
                    r[key]
                    for r in results
                ]
            )
        )

    return {

        "name":
            name,

        "rms":
            mean(
                "rms"
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
                "episode_length"
            ),

        "termination_rate":
            float(
                np.mean(
                    [
                        r[
                            "terminated"
                        ]
                        for r in results
                    ]
                )
                * 100.0
            ),

        "absolute_limit_fraction":
            mean(
                "absolute_limit_fraction"
            )
    }


v3 = summarize(
    "PPO v3",
    v3_results
)

v4 = summarize(
    "PPO v4",
    v4_results
)


print(
    "\n========== PPO V3 vs V4 =========="
)


for result in [
    v3,
    v4
]:

    print(
        f"\n--- {result['name']} ---"
    )

    print(
        f"Mean RMS error          : "
        f"{result['rms']:.6f} rad"
    )

    print(
        f"Mean peak torque        : "
        f"{result['peak_torque']:.6f} N*m"
    )

    print(
        f"Mean control effort     : "
        f"{result['control_effort']:.3f} N²*m²*s"
    )

    print(
        f"Mean max torque step    : "
        f"{result['max_torque_step']:.6f} N*m"
    )

    print(
        f"Mean abs torque step    : "
        f"{result['mean_abs_torque_step']:.6f} N*m"
    )

    print(
        f"Episode length          : "
        f"{result['episode_length']:.2f} / 200"
    )

    print(
        f"Termination rate        : "
        f"{result['termination_rate']:.1f}%"
    )

    print(
        f"Absolute limit active   : "
        f"{result['absolute_limit_fraction']:.2f}%"
    )


# ============================================================
# Training curve
# ============================================================

episode_rewards = []

episode_lengths = []


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
                row["r"]
            )
        )

        episode_lengths.append(
            int(
                row["l"]
            )
        )


episode_rewards = np.asarray(
    episode_rewards
)

episode_lengths = np.asarray(
    episode_lengths
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

x_ma = np.arange(
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
    x_ma,
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
    "PPO V4 Learning Curve"
)

plt.legend()
plt.grid(True)
plt.tight_layout()


learning_fig = os.path.join(
    RESULT_DIR,
    "ppo_v4_learning_curve.png"
)

plt.savefig(
    learning_fig,
    dpi=200
)


# ============================================================
# Tracking
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    v4_trace[
        "time"
    ],

    v4_trace[
        "qd"
    ],

    linestyle="--",
    label="Desired trajectory"
)

plt.plot(
    v3_trace[
        "time"
    ],

    v3_trace[
        "q"
    ],

    label="PPO v3"
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

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Joint 2 Position (rad)"
)

plt.title(
    "PPO V3 vs V4: Tracking"
)

plt.legend()
plt.grid(True)
plt.tight_layout()


tracking_fig = os.path.join(
    RESULT_DIR,
    "ppo_v3_vs_v4_tracking.png"
)

plt.savefig(
    tracking_fig,
    dpi=200
)


# ============================================================
# Torque
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    v3_trace[
        "time"
    ],

    v3_trace[
        "torque"
    ],

    label="PPO v3"
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

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Torque (N*m)"
)

plt.title(
    "PPO V3 vs V4: Torque Command"
)

plt.legend()
plt.grid(True)
plt.tight_layout()


torque_fig = os.path.join(
    RESULT_DIR,
    "ppo_v3_vs_v4_torque.png"
)

plt.savefig(
    torque_fig,
    dpi=200
)


# ============================================================
# Torque step
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    v3_trace[
        "time"
    ],

    np.abs(
        v3_trace[
            "torque_step"
        ]
    ),

    label="PPO v3"
)

plt.plot(
    v4_trace[
        "time"
    ],

    np.abs(
        v4_trace[
            "torque_step"
        ]
    ),

    label="PPO v4"
)

plt.axhline(
    10.0,
    linestyle="--",
    label="PPO v4 hard rate limit"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "|Torque Step| (N*m)"
)

plt.title(
    "PPO V3 vs V4: Torque-Rate Behavior"
)

plt.legend()
plt.grid(True)
plt.tight_layout()


step_fig = os.path.join(
    RESULT_DIR,
    "ppo_v3_vs_v4_torque_step.png"
)

plt.savefig(
    step_fig,
    dpi=200
)


print(
    "\n[PASS] PPO v4 comparison finished"
)

print(
    "\nFigures saved to:"
)

print(
    learning_fig
)

print(
    tracking_fig
)

print(
    torque_fig
)

print(
    step_fig
)
