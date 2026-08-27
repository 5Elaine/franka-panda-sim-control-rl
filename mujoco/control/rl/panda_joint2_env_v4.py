import os
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces

import mujoco
import numpy as np


class PandaJoint2EnvV4(gym.Env):

    metadata = {
        "render_modes": []
    }

    def __init__(self):

        super().__init__()

        # ====================================================
        # Model
        # ====================================================

        project_root = Path(__file__).resolve().parents[3]
        menagerie_root = Path(os.environ.get(
            "MUJOCO_MENAGERIE_PATH",
            project_root / "third_party" / "mujoco_menagerie",
        )).expanduser()
        self.model_path = str(
            menagerie_root / "franka_emika_panda" / "scene.xml"
        )

        self.joint_id = 1
        self.actuator_id = 1

        # ====================================================
        # Actuator constraints
        # ====================================================

        self.torque_limit = 60.0

        # HARD torque-rate constraint:
        #
        # maximum change every 20 ms
        self.max_torque_step = 10.0

        # ====================================================
        # Timing
        # ====================================================

        self.control_dt = 0.020
        self.episode_time = 4.0

        # ====================================================
        # Reference trajectory
        # ====================================================

        self.center = -0.5
        self.amplitude = 0.25
        self.frequency = 0.5

        self.omega = (
            2.0
            * np.pi
            * self.frequency
        )

        # ====================================================
        # MuJoCo
        # ====================================================

        self.model = mujoco.MjModel.from_xml_path(
            self.model_path
        )

        self._configure_torque_actuator()

        self.data = mujoco.MjData(
            self.model
        )

        self.physics_dt = float(
            self.model.opt.timestep
        )

        self.physics_steps_per_control = int(
            round(
                self.control_dt
                / self.physics_dt
            )
        )

        self.max_episode_steps = int(
            self.episode_time
            / self.control_dt
        )

        # ====================================================
        # ACTION
        #
        # This time action is NOT direct torque.
        #
        # action [-1, 1]
        #       ↓
        # delta torque [-10, +10] N*m
        # ====================================================

        self.action_space = spaces.Box(

            low=np.array(
                [-1.0],
                dtype=np.float32
            ),

            high=np.array(
                [1.0],
                dtype=np.float32
            ),

            dtype=np.float32
        )

        # ====================================================
        # OBSERVATION
        #
        # [
        # q_normalized,
        # dq_normalized,
        # position_error_normalized,
        # velocity_error_normalized,
        # current_torque_normalized,
        # sin(phase),
        # cos(phase),
        # remaining_time_fraction
        # ]
        # ====================================================

        self.observation_space = spaces.Box(

            low=np.array(
                [
                    -1.2,
                    -2.5,
                    -5.0,
                    -5.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    0.0
                ],
                dtype=np.float32
            ),

            high=np.array(
                [
                    1.2,
                    2.5,
                    5.0,
                    5.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0
                ],
                dtype=np.float32
            ),

            dtype=np.float32
        )

        self.step_count = 0

        self.current_torque = 0.0


    # ========================================================
    # Reference
    # ========================================================

    def desired_trajectory(self, t):

        phase = (
            self.omega
            * t
        )

        q_des = (
            self.center
            + self.amplitude
            * np.sin(
                phase
            )
        )

        dq_des = (
            self.amplitude
            * self.omega
            * np.cos(
                phase
            )
        )

        return (
            q_des,
            dq_des,
            phase
        )


    # ========================================================
    # Torque actuator
    # ========================================================

    def _configure_torque_actuator(self):

        aid = self.actuator_id

        self.model.actuator_gainprm[
            aid, :
        ] = 0.0

        self.model.actuator_gainprm[
            aid, 0
        ] = 1.0

        self.model.actuator_biasprm[
            aid, :
        ] = 0.0

        self.model.actuator_ctrlrange[
            aid
        ] = [
            -self.torque_limit,
            self.torque_limit
        ]

        self.model.actuator_ctrllimited[
            aid
        ] = 1

        self.model.actuator_forcerange[
            aid
        ] = [
            -self.torque_limit,
            self.torque_limit
        ]

        self.model.actuator_forcelimited[
            aid
        ] = 1


    # ========================================================
    # State
    # ========================================================

    def _get_state_terms(self):

        t = float(
            self.data.time
        )

        q = float(
            self.data.qpos[
                self.joint_id
            ]
        )

        dq = float(
            self.data.qvel[
                self.joint_id
            ]
        )

        (
            q_des,
            dq_des,
            phase
        ) = self.desired_trajectory(
            t
        )

        e = (
            q
            - q_des
        )

        edot = (
            dq
            - dq_des
        )

        return (
            t,
            q,
            dq,
            q_des,
            dq_des,
            phase,
            e,
            edot
        )


    # ========================================================
    # Observation
    # ========================================================

    def _get_observation(self):

        (
            t,
            q,
            dq,
            q_des,
            dq_des,
            phase,
            e,
            edot
        ) = self._get_state_terms()

        remaining_fraction = np.clip(
            (
                self.episode_time
                - t
            )
            / self.episode_time,
            0.0,
            1.0
        )

        return np.array(
            [
                q / 1.70,

                dq / 4.00,

                e / 0.50,

                edot / 2.00,

                self.current_torque
                / self.torque_limit,

                np.sin(
                    phase
                ),

                np.cos(
                    phase
                ),

                remaining_fraction
            ],
            dtype=np.float32
        )


    # ========================================================
    # Reward
    #
    # IMPORTANT:
    # reward evaluates the state AFTER action execution.
    # ========================================================

    def _calculate_reward(
        self,
        applied_torque,
        applied_delta_torque
    ):

        (
            t,
            q,
            dq,
            q_des,
            dq_des,
            phase,
            e,
            edot
        ) = self._get_state_terms()

        position_score = np.exp(
            -(
                e / 0.12
            ) ** 2
        )

        velocity_score = np.exp(
            -(
                edot / 0.80
            ) ** 2
        )

        torque_cost = (
            applied_torque
            / self.torque_limit
        ) ** 2

        torque_change_cost = (
            applied_delta_torque
            / self.max_torque_step
        ) ** 2

        reward = (
            1.00
            * position_score

            + 0.20
            * velocity_score

            - 0.02
            * torque_cost

            - 0.03
            * torque_change_cost
        )

        return float(
            reward
        )


    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(
            seed=seed
        )

        mujoco.mj_resetData(
            self.model,
            self.data
        )

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        (
            q_des_0,
            dq_des_0,
            _
        ) = self.desired_trajectory(
            0.0
        )

        q_noise = self.np_random.uniform(
            -0.05,
            0.05
        )

        dq_noise = self.np_random.uniform(
            -0.10,
            0.10
        )

        self.data.qpos[
            self.joint_id
        ] = (
            q_des_0
            + q_noise
        )

        self.data.qvel[
            self.joint_id
        ] = (
            dq_des_0
            + dq_noise
        )

        self.current_torque = 0.0

        self.data.ctrl[
            self.actuator_id
        ] = self.current_torque

        mujoco.mj_forward(
            self.model,
            self.data
        )

        self.step_count = 0

        return (
            self._get_observation(),
            {}
        )


    # ========================================================
    # Step
    # ========================================================

    def step(
        self,
        action
    ):

        # ----------------------------------------------------
        # Policy action
        #
        # [-1, 1]
        #      ↓
        # desired torque increment
        # ----------------------------------------------------

        normalized_action = float(
            np.clip(
                action[0],
                -1.0,
                1.0
            )
        )

        requested_delta_torque = (
            normalized_action
            * self.max_torque_step
        )

        previous_torque = (
            self.current_torque
        )

        # ----------------------------------------------------
        # HARD rate constraint is intrinsic:
        #
        # requested_delta_torque is already
        # within ±10 N*m.
        # ----------------------------------------------------

        requested_torque = (
            previous_torque
            + requested_delta_torque
        )

        # ----------------------------------------------------
        # HARD absolute torque limit
        # ----------------------------------------------------

        self.current_torque = float(
            np.clip(
                requested_torque,
                -self.torque_limit,
                self.torque_limit
            )
        )

        actual_delta_torque = (
            self.current_torque
            - previous_torque
        )

        # ----------------------------------------------------
        # Apply torque
        # ----------------------------------------------------

        self.data.ctrl[
            self.actuator_id
        ] = self.current_torque

        self.data.ctrl[0] = 0.0
        self.data.ctrl[2:7] = 0.0

        for _ in range(
            self.physics_steps_per_control
        ):

            mujoco.mj_step(
                self.model,
                self.data
            )

        self.step_count += 1

        # ----------------------------------------------------
        # Reward AFTER action execution
        # ----------------------------------------------------

        reward = self._calculate_reward(
            self.current_torque,
            actual_delta_torque
        )

        (
            t,
            q,
            dq,
            q_des,
            dq_des,
            phase,
            e,
            edot
        ) = self._get_state_terms()

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        unsafe = (
            not np.isfinite(q)
            or not np.isfinite(dq)
            or abs(q) > 1.70
            or abs(dq) > 8.0
        )

        terminated = bool(
            unsafe
        )

        if terminated:

            reward -= 10.0

        truncated = bool(
            self.step_count
            >= self.max_episode_steps
        )

        info = {

            "time":
                float(t),

            "q":
                float(q),

            "dq":
                float(dq),

            "q_des":
                float(q_des),

            "dq_des":
                float(dq_des),

            "position_error":
                float(e),

            "velocity_error":
                float(edot),

            "normalized_action":
                normalized_action,

            "torque":
                float(
                    self.current_torque
                ),

            "torque_change":
                float(
                    actual_delta_torque
                ),

            "absolute_limit_active":
                bool(
                    abs(
                        requested_torque
                    )
                    >
                    self.torque_limit
                )
        }

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            info
        )


    def close(self):
        pass
