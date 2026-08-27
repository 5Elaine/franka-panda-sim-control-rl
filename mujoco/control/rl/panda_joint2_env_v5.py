import os
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces

import mujoco
import numpy as np


class PandaJoint2EnvV5(gym.Env):

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

        self.model = mujoco.MjModel.from_xml_path(
            self.model_path
        )

        self.data = mujoco.MjData(
            self.model
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # joint ID and DOF ID are not conceptually identical.
        # Use jnt_dofadr to locate the joint's velocity/force DOF.
        # ----------------------------------------------------

        self.dof_id = int(
            self.model.jnt_dofadr[
                self.joint_id
            ]
        )

        # ====================================================
        # Save NOMINAL dynamics
        #
        # Every reset will randomize relative to THIS value.
        # Never randomize relative to the previous episode.
        # ====================================================

        self.nominal_damping = float(
            self.model.dof_damping[
                self.dof_id
            ]
        )

        # ====================================================
        # Control constraints
        #
        # These are NOT randomized.
        # ====================================================

        self.torque_limit = 60.0

        self.max_torque_step = 10.0

        # ====================================================
        # Timing
        # ====================================================

        self.control_dt = 0.020
        self.episode_time = 4.0

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
        # Randomization ranges
        # ====================================================

        # Task randomization
        self.frequency_range = (
            0.35,
            0.65
        )

        self.amplitude_range = (
            0.20,
            0.30
        )

        self.center_range = (
            -0.60,
            -0.40
        )

        # Initial-state randomization
        self.initial_position_offset_range = (
            -0.15,
            0.15
        )

        self.initial_velocity_offset_range = (
            -0.50,
            0.50
        )

        # Domain randomization
        self.damping_scale_range = (
            0.80,
            1.20
        )

        self.external_torque_range = (
            -5.0,
            5.0
        )

        # ====================================================
        # Current randomized episode parameters
        # ====================================================

        self.frequency = 0.50
        self.amplitude = 0.25
        self.center = -0.50

        self.omega = (
            2.0
            * np.pi
            * self.frequency
        )

        self.damping_scale = 1.0
        self.external_torque = 0.0

        self.initial_position_offset = 0.0
        self.initial_velocity_offset = 0.0

        # ====================================================
        # Torque actuator
        # ====================================================

        self._configure_torque_actuator()

        # ====================================================
        # Action
        #
        # action ∈ [-1,1]
        #
        # action means delta torque,
        # NOT direct torque.
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
        # Observation: 11 dimensions
        #
        # [
        #   q,
        #   dq,
        #   position error,
        #   velocity error,
        #   current torque,
        #   sin phase,
        #   cos phase,
        #   remaining time,
        #   task frequency,
        #   task amplitude,
        #   task center
        # ]
        #
        # All approximately normalized.
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
                    0.0,
                    -1.0,
                    -1.0,
                    -1.0,
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
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                dtype=np.float32
            ),

            dtype=np.float32
        )

        self.step_count = 0

        self.current_torque = 0.0


    # ========================================================
    # Configure direct torque actuator
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
    # Reference trajectory
    # ========================================================

    def desired_trajectory(
        self,
        t
    ):

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
                self.dof_id
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

        # ----------------------------------------------------
        # Normalize task parameters to approximately [-1,1].
        # ----------------------------------------------------

        frequency_norm = (
            self.frequency
            - 0.50
        ) / 0.15

        amplitude_norm = (
            self.amplitude
            - 0.25
        ) / 0.05

        center_norm = (
            self.center
            + 0.50
        ) / 0.10

        observation = np.array(
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

                remaining_fraction,

                frequency_norm,

                amplitude_norm,

                center_norm,
            ],
            dtype=np.float32
        )

        return observation


    # ========================================================
    # Reward
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
    #
    # A new random "world" is sampled here.
    # ========================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(
            seed=seed
        )

        # ----------------------------------------------------
        # Reset MuJoCo dynamic data
        # ----------------------------------------------------

        mujoco.mj_resetData(
            self.model,
            self.data
        )

        # ----------------------------------------------------
        # Restore NOMINAL dynamics first.
        # ----------------------------------------------------

        self.model.dof_damping[
            self.dof_id
        ] = (
            self.nominal_damping
        )

        # ====================================================
        # 1. TASK RANDOMIZATION
        # ====================================================

        self.frequency = float(
            self.np_random.uniform(
                *self.frequency_range
            )
        )

        self.amplitude = float(
            self.np_random.uniform(
                *self.amplitude_range
            )
        )

        self.center = float(
            self.np_random.uniform(
                *self.center_range
            )
        )

        self.omega = (
            2.0
            * np.pi
            * self.frequency
        )

        # ====================================================
        # 2. DYNAMICS RANDOMIZATION
        # ====================================================

        self.damping_scale = float(
            self.np_random.uniform(
                *self.damping_scale_range
            )
        )

        self.model.dof_damping[
            self.dof_id
        ] = (
            self.nominal_damping
            * self.damping_scale
        )

        # ====================================================
        # 3. INITIAL-STATE RANDOMIZATION
        # ====================================================

        (
            q_des_0,
            dq_des_0,
            _
        ) = self.desired_trajectory(
            0.0
        )

        self.initial_position_offset = float(
            self.np_random.uniform(
                *self.initial_position_offset_range
            )
        )

        self.initial_velocity_offset = float(
            self.np_random.uniform(
                *self.initial_velocity_offset_range
            )
        )

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0

        self.data.qpos[
            self.joint_id
        ] = (
            q_des_0
            + self.initial_position_offset
        )

        self.data.qvel[
            self.dof_id
        ] = (
            dq_des_0
            + self.initial_velocity_offset
        )

        # ====================================================
        # 4. DISTURBANCE RANDOMIZATION
        # ====================================================

        self.external_torque = float(
            self.np_random.uniform(
                *self.external_torque_range
            )
        )

        self.data.qfrc_applied[
            self.dof_id
        ] = (
            self.external_torque
        )

        # ====================================================
        # Reset control state
        # ====================================================

        self.current_torque = 0.0

        self.data.ctrl[
            self.actuator_id
        ] = 0.0

        self.step_count = 0

        mujoco.mj_forward(
            self.model,
            self.data
        )

        info = {
            "frequency":
                self.frequency,

            "amplitude":
                self.amplitude,

            "center":
                self.center,

            "initial_position_offset":
                self.initial_position_offset,

            "initial_velocity_offset":
                self.initial_velocity_offset,

            "external_torque":
                self.external_torque,

            "damping_scale":
                self.damping_scale,

            "actual_damping":
                float(
                    self.model.dof_damping[
                        self.dof_id
                    ]
                )
        }

        return (
            self._get_observation(),
            info
        )


    # ========================================================
    # Step
    # ========================================================

    def step(
        self,
        action
    ):

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

        requested_torque = (
            previous_torque
            + requested_delta_torque
        )

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

        self.data.ctrl[
            self.actuator_id
        ] = (
            self.current_torque
        )

        self.data.ctrl[0] = 0.0
        self.data.ctrl[2:7] = 0.0

        # Keep the episode-specific disturbance active.
        self.data.qfrc_applied[
            self.dof_id
        ] = (
            self.external_torque
        )

        for _ in range(
            self.physics_steps_per_control
        ):

            mujoco.mj_step(
                self.model,
                self.data
            )

        self.step_count += 1

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

            "torque":
                float(
                    self.current_torque
                ),

            "torque_change":
                float(
                    actual_delta_torque
                ),

            "frequency":
                self.frequency,

            "amplitude":
                self.amplitude,

            "center":
                self.center,

            "external_torque":
                self.external_torque,

            "damping_scale":
                self.damping_scale,
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
