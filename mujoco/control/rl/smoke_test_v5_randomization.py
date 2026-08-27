import numpy as np

from stable_baselines3.common.env_checker import (
    check_env
)

from panda_joint2_env_v5 import (
    PandaJoint2EnvV5
)


print(
    "========== PPO V5 RANDOMIZATION CHECK =========="
)

env = PandaJoint2EnvV5()

check_env(
    env,
    warn=True
)

print(
    "[PASS] Gymnasium / SB3 environment check passed"
)

print()
print(
    "Nominal joint damping :",
    env.nominal_damping
)

print(
    "DOF ID                :",
    env.dof_id
)

print(
    "Observation dimension :",
    env.observation_space.shape
)

print(
    "Action dimension      :",
    env.action_space.shape
)


records = []


for episode in range(
    100
):

    obs, info = env.reset(
        seed=10000 + episode
    )

    records.append(
        [
            info[
                "frequency"
            ],

            info[
                "amplitude"
            ],

            info[
                "center"
            ],

            info[
                "initial_position_offset"
            ],

            info[
                "initial_velocity_offset"
            ],

            info[
                "external_torque"
            ],

            info[
                "damping_scale"
            ],

            info[
                "actual_damping"
            ],
        ]
    )


records = np.asarray(
    records
)


names = [
    "Frequency [Hz]",
    "Amplitude [rad]",
    "Center [rad]",
    "Initial q offset [rad]",
    "Initial dq offset [rad/s]",
    "External torque [N*m]",
    "Damping scale",
    "Actual damping",
]


print()
print(
    "========== SAMPLED RANGE OVER 100 EPISODES =========="
)


for i, name in enumerate(
    names
):

    values = records[
        :,
        i
    ]

    print(
        f"{name:26s}: "
        f"min={np.min(values): .6f}   "
        f"max={np.max(values): .6f}   "
        f"mean={np.mean(values): .6f}"
    )


# ============================================================
# Hard assertions
# ============================================================

assert np.all(
    (
        records[:, 0] >= 0.35
    )
    &
    (
        records[:, 0] <= 0.65
    )
)

assert np.all(
    (
        records[:, 1] >= 0.20
    )
    &
    (
        records[:, 1] <= 0.30
    )
)

assert np.all(
    (
        records[:, 2] >= -0.60
    )
    &
    (
        records[:, 2] <= -0.40
    )
)

assert np.all(
    (
        records[:, 3] >= -0.15
    )
    &
    (
        records[:, 3] <= 0.15
    )
)

assert np.all(
    (
        records[:, 4] >= -0.50
    )
    &
    (
        records[:, 4] <= 0.50
    )
)

assert np.all(
    (
        records[:, 5] >= -5.0
    )
    &
    (
        records[:, 5] <= 5.0
    )
)

assert np.all(
    (
        records[:, 6] >= 0.80
    )
    &
    (
        records[:, 6] <= 1.20
    )
)


# ============================================================
# Test one random episode with random actions
#
# Not testing tracking quality.
# Just testing that all interfaces still work.
# ============================================================

obs, info = env.reset(
    seed=42
)

terminated = False
truncated = False

max_torque_step = 0.0

steps = 0


while not (
    terminated
    or truncated
):

    action = env.action_space.sample()

    (
        obs,
        reward,
        terminated,
        truncated,
        step_info
    ) = env.step(
        action
    )

    max_torque_step = max(
        max_torque_step,
        abs(
            step_info[
                "torque_change"
            ]
        )
    )

    steps += 1


print()
print(
    "========== RANDOM ACTION EPISODE =========="
)

print(
    "Episode steps          :",
    steps
)

print(
    "Terminated             :",
    terminated
)

print(
    "Truncated              :",
    truncated
)

print(
    "Maximum torque step    :",
    f"{max_torque_step:.6f} N*m"
)

assert (
    max_torque_step
    <= 10.000001
)


env.close()


print()
print(
    "[PASS] PPO v5 randomization smoke test finished"
)
