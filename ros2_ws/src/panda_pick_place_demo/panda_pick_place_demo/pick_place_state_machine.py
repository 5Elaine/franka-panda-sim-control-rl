#!/usr/bin/env python3

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Quaternion

from panda_pick_place_demo.preflight_checker import PreflightChecker
from panda_pick_place_demo.scene_manager import SceneManager
from panda_pick_place_demo.gripper_manager import GripperManager
from panda_pick_place_demo.motion_manager import MotionManager


HOME_JOINTS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.785,
    "panda_joint3": 0.0,
    "panda_joint4": -2.356,
    "panda_joint5": 0.0,
    "panda_joint6": 1.571,
    "panda_joint7": 0.785,
}


class TaskState(Enum):
    INITIALIZE_SCENE = auto()
    OPEN_GRIPPER = auto()
    MOVE_TO_PREGRASP = auto()
    CARTESIAN_APPROACH = auto()
    CLOSE_GRIPPER = auto()
    ATTACH_CUBE = auto()
    LIFT_CUBE = auto()
    TRANSPORT_CUBE = auto()
    LOWER_TO_PREPLACE = auto()
    DETACH_CUBE = auto()
    RELEASE_CUBE = auto()
    UPDATE_PLACED_SCENE = auto()
    RETREAT = auto()
    RETURN_HOME = auto()
    COMPLETE = auto()


@dataclass(frozen=True)
class StateDescription:
    state: TaskState
    label: str
    purpose: str


TASK_SEQUENCE = [
    StateDescription(
        TaskState.INITIALIZE_SCENE,
        "INITIALIZE_SCENE",
        "Add the table and cube to the MoveIt Planning Scene.",
    ),
    StateDescription(
        TaskState.OPEN_GRIPPER,
        "OPEN_GRIPPER",
        "Open the gripper to the safe position 0.035 m.",
    ),
    StateDescription(
        TaskState.MOVE_TO_PREGRASP,
        "MOVE_TO_PREGRASP",
        "Use OMPL to reach the pose above the cube.",
    ),
    StateDescription(
        TaskState.CARTESIAN_APPROACH,
        "CARTESIAN_APPROACH",
        "Descend in two verified Cartesian segments to the grasp pose.",
    ),
    StateDescription(
        TaskState.CLOSE_GRIPPER,
        "CLOSE_GRIPPER",
        "Close the fingers around the cube.",
    ),
    StateDescription(
        TaskState.ATTACH_CUBE,
        "ATTACH_CUBE",
        "Mark the cube as attached in the Planning Scene.",
    ),
    StateDescription(
        TaskState.LIFT_CUBE,
        "LIFT_CUBE",
        "Lift the cube vertically to transport height.",
    ),
    StateDescription(
        TaskState.TRANSPORT_CUBE,
        "TRANSPORT_CUBE",
        "Use OMPL to move the attached cube to the place area.",
    ),
    StateDescription(
        TaskState.LOWER_TO_PREPLACE,
        "LOWER_TO_PREPLACE",
        "Lower the cube collision-aware toward the table.",
    ),
    StateDescription(
        TaskState.DETACH_CUBE,
        "DETACH_CUBE",
        "Remove the cube from the attached-object list.",
    ),
    StateDescription(
        TaskState.RELEASE_CUBE,
        "RELEASE_CUBE",
        "Open the gripper and physically release the cube.",
    ),
    StateDescription(
        TaskState.UPDATE_PLACED_SCENE,
        "UPDATE_PLACED_SCENE",
        "Update the cube to its new world pose in MoveIt.",
    ),
    StateDescription(
        TaskState.RETREAT,
        "RETREAT",
        "Move the open gripper vertically away from the cube.",
    ),
    StateDescription(
        TaskState.RETURN_HOME,
        "RETURN_HOME",
        "Use OMPL to return safely to the Home joint state.",
    ),
    StateDescription(
        TaskState.COMPLETE,
        "COMPLETE",
        "Report successful completion of the task.",
    ),
]


class PickPlaceStateMachine(Node):
    def __init__(self) -> None:
        super().__init__("panda_pick_place_state_machine")

        self.declare_parameter("dry_run", True)
        self.declare_parameter(
            "release_settle_time",
            2.0,
        )
        self.declare_parameter(
            "retreat_distance",
            0.050,
        )
        self.declare_parameter(
            "open_gripper_position",
            0.035,
        )
        self.declare_parameter(
            "close_gripper_position",
            0.024,
        )
        self.declare_parameter(
            "stop_after_state",
            "INITIALIZE_SCENE",
        )

        self.declare_parameter("pick_x", 0.55)
        self.declare_parameter("pick_y", 0.00)
        self.declare_parameter("cube_z", 0.425)

        self.declare_parameter("place_x", 0.55)
        self.declare_parameter("place_y", 0.25)

        self.declare_parameter(
            "grasp_orientation_x",
            1.0,
        )
        self.declare_parameter(
            "grasp_orientation_y",
            0.0,
        )
        self.declare_parameter(
            "grasp_orientation_z",
            0.0,
        )
        self.declare_parameter(
            "grasp_orientation_w",
            0.0,
        )

        self.declare_parameter("pregrasp_z", 0.580)
        self.declare_parameter("approach_tcp_z", 0.500)
        self.declare_parameter("grasp_tcp_z", 0.455)
        self.declare_parameter("transport_tcp_z", 0.505)
        self.declare_parameter("preplace_tcp_z", 0.467)
        self.declare_parameter(
            "micro_lift_distance",
            0.010,
        )

        self.dry_run = bool(
            self.get_parameter("dry_run").value
        )

        self.stop_after_state = str(
            self.get_parameter("stop_after_state").value
        )

        self.scene_manager = SceneManager(self)
        self.gripper_manager = GripperManager(self)
        self.motion_manager = MotionManager(self)

    def print_task_parameters(self) -> None:
        pick_x = float(self.get_parameter("pick_x").value)
        pick_y = float(self.get_parameter("pick_y").value)
        cube_z = float(self.get_parameter("cube_z").value)

        place_x = float(
            self.get_parameter("place_x").value
        )
        place_y = float(
            self.get_parameter("place_y").value
        )

        pregrasp_z = float(
            self.get_parameter("pregrasp_z").value
        )
        approach_tcp_z = float(
            self.get_parameter("approach_tcp_z").value
        )
        grasp_tcp_z = float(
            self.get_parameter("grasp_tcp_z").value
        )
        transport_tcp_z = float(
            self.get_parameter("transport_tcp_z").value
        )
        preplace_tcp_z = float(
            self.get_parameter("preplace_tcp_z").value
        )

        print()
        print("=== AUTOMATED PICK-AND-PLACE PARAMETERS ===")
        print(
            "Pick cube centre:  "
            f"({pick_x:.3f}, {pick_y:.3f}, {cube_z:.3f}) m"
        )
        print(
            "Place cube centre: "
            f"({place_x:.3f}, {place_y:.3f}, {cube_z:.3f}) m"
        )
        print(f"Pregrasp TCP z:   {pregrasp_z:.3f} m")
        print(
            f"Approach TCP z:   {approach_tcp_z:.3f} m"
        )
        print(f"Grasp TCP z:      {grasp_tcp_z:.3f} m")
        print(
            f"Transport TCP z:  {transport_tcp_z:.3f} m"
        )
        print(
            f"Pre-place TCP z:  {preplace_tcp_z:.3f} m"
        )

    def execute_state(
        self,
        description: StateDescription,
    ) -> bool:
        if description.state == TaskState.INITIALIZE_SCENE:
            return self.scene_manager.initialize_pick_scene()

        if description.state == TaskState.OPEN_GRIPPER:
            open_position = float(
                self.get_parameter(
                    "open_gripper_position"
                ).value
            )

            return self.gripper_manager.command(
                position=open_position,
                max_effort=0.0,
            )

        if description.state == TaskState.MOVE_TO_PREGRASP:
            pick_x = float(
                self.get_parameter("pick_x").value
            )
            pick_y = float(
                self.get_parameter("pick_y").value
            )
            pregrasp_z = float(
                self.get_parameter("pregrasp_z").value
            )

            grasp_orientation = Quaternion()
            grasp_orientation.x = float(
                self.get_parameter(
                    "grasp_orientation_x"
                ).value
            )
            grasp_orientation.y = float(
                self.get_parameter(
                    "grasp_orientation_y"
                ).value
            )
            grasp_orientation.z = float(
                self.get_parameter(
                    "grasp_orientation_z"
                ).value
            )
            grasp_orientation.w = float(
                self.get_parameter(
                    "grasp_orientation_w"
                ).value
            )

            self.get_logger().info(
                "Using fixed top-down grasp orientation: "
                f"q=({grasp_orientation.x:.3f}, "
                f"{grasp_orientation.y:.3f}, "
                f"{grasp_orientation.z:.3f}, "
                f"{grasp_orientation.w:.3f})"
            )

            return self.motion_manager.move_to_pose_ompl(
                target_x=pick_x,
                target_y=pick_y,
                target_z=pregrasp_z,
                constraint_name=(
                    "pregrasp_above_test_cube"
                ),
                velocity_scaling=0.20,
                acceleration_scaling=0.20,
                target_orientation=grasp_orientation,
                orientation_tolerance=0.08,
            )

        if description.state == TaskState.CARTESIAN_APPROACH:
            pick_x = float(
                self.get_parameter("pick_x").value
            )
            pick_y = float(
                self.get_parameter("pick_y").value
            )
            approach_z = float(
                self.get_parameter(
                    "approach_tcp_z"
                ).value
            )
            grasp_z = float(
                self.get_parameter(
                    "grasp_tcp_z"
                ).value
            )

            first_segment_ok = (
                self.motion_manager.move_cartesian_to(
                    target_x=pick_x,
                    target_y=pick_y,
                    target_z=approach_z,
                    motion_label=(
                        "Cartesian approach segment 1/2"
                    ),
                )
            )

            if not first_segment_ok:
                return False

            return self.motion_manager.move_cartesian_to(
                target_x=pick_x,
                target_y=pick_y,
                target_z=grasp_z,
                motion_label=(
                    "Cartesian near-grasp segment 2/2"
                ),
            )

        if description.state == TaskState.CLOSE_GRIPPER:
            close_position = float(
                self.get_parameter(
                    "close_gripper_position"
                ).value
            )

            command_ok = self.gripper_manager.command(
                position=close_position,
                max_effort=0.0,
                position_tolerance=0.003,
            )

            if not command_ok:
                return False

            # The Action may report success while the fingers
            # are still converging inside its goal tolerance.
            # Wait for the measured joint state before Attach.
            return (
                self.gripper_manager.wait_for_grasp_closure(
                    maximum_position=0.0252,
                    timeout_sec=4.0,
                    stable_duration=0.30,
                )
            )

        if description.state == TaskState.ATTACH_CUBE:
            return self.scene_manager.attach_cube_to_hand()

        if description.state == TaskState.LIFT_CUBE:
            pick_x = float(
                self.get_parameter("pick_x").value
            )
            pick_y = float(
                self.get_parameter("pick_y").value
            )
            transport_z = float(
                self.get_parameter(
                    "transport_tcp_z"
                ).value
            )
            micro_distance = float(
                self.get_parameter(
                    "micro_lift_distance"
                ).value
            )

            # Use the measured position for the first small lift.
            current_transform = (
                self.motion_manager.lookup_tcp_transform()
            )
            current_position = (
                current_transform.transform.translation
            )

            micro_target_z = (
                current_position.z + micro_distance
            )

            self.get_logger().info(
                "Starting physical grasp verification lift: "
                f"z={current_position.z:.3f} -> "
                f"{micro_target_z:.3f} m"
            )

            micro_lift_ok = (
                self.motion_manager.move_cartesian_to(
                    target_x=current_position.x,
                    target_y=current_position.y,
                    target_z=micro_target_z,
                    motion_label=(
                        "1 cm physical grasp verification lift"
                    ),
                    slowdown_factor=5.0,
                    position_tolerance=0.004,
                    skip_tolerance=0.001,
                )
            )

            if not micro_lift_ok:
                self.get_logger().error(
                    "Micro-lift failed. The transport lift "
                    "will not be executed."
                )
                return False

            self.get_logger().info(
                "Micro-lift trajectory succeeded. "
                "Check that the Gazebo cube left the table."
            )

            return self.motion_manager.move_cartesian_to(
                target_x=pick_x,
                target_y=pick_y,
                target_z=transport_z,
                motion_label=(
                    "Lift attached cube to transport height"
                ),
                slowdown_factor=5.0,
                position_tolerance=0.015,
            )

        if description.state == TaskState.TRANSPORT_CUBE:
            place_x = float(
                self.get_parameter("place_x").value
            )
            place_y = float(
                self.get_parameter("place_y").value
            )
            transport_z = float(
                self.get_parameter(
                    "transport_tcp_z"
                ).value
            )

            if not self.scene_manager.verify_cube_attached():
                self.get_logger().error(
                    "Transport blocked because the cube "
                    "attachment state is invalid."
                )
                return False

            grasp_orientation = Quaternion()
            grasp_orientation.x = float(
                self.get_parameter(
                    "grasp_orientation_x"
                ).value
            )
            grasp_orientation.y = float(
                self.get_parameter(
                    "grasp_orientation_y"
                ).value
            )
            grasp_orientation.z = float(
                self.get_parameter(
                    "grasp_orientation_z"
                ).value
            )
            grasp_orientation.w = float(
                self.get_parameter(
                    "grasp_orientation_w"
                ).value
            )

            transport_ok = (
                self.motion_manager.move_to_pose_ompl(
                    target_x=place_x,
                    target_y=place_y,
                    target_z=transport_z,
                    constraint_name=(
                        "transport_attached_cube"
                    ),
                    velocity_scaling=0.10,
                    acceleration_scaling=0.10,
                    target_orientation=grasp_orientation,
                    orientation_tolerance=0.08,
                )
            )

            if not transport_ok:
                return False

            # Confirm that MoveIt still considers the cube
            # attached after executing the transport motion.
            if not self.scene_manager.verify_cube_attached():
                self.get_logger().error(
                    "The cube attachment state was lost "
                    "during transport."
                )
                return False

            return True

        if description.state == TaskState.LOWER_TO_PREPLACE:
            if not self.scene_manager.verify_cube_attached():
                self.get_logger().error(
                    "Pre-place descent blocked because the "
                    "cube attachment state is invalid."
                )
                return False

            preplace_z = float(
                self.get_parameter(
                    "preplace_tcp_z"
                ).value
            )

            current_transform = (
                self.motion_manager.lookup_tcp_transform()
            )
            current_position = (
                current_transform.transform.translation
            )

            self.get_logger().info(
                "Preparing vertical pre-place descent: "
                f"z={current_position.z:.3f} -> "
                f"{preplace_z:.3f} m"
            )

            descent_ok = (
                self.motion_manager.move_cartesian_to(
                    target_x=current_position.x,
                    target_y=current_position.y,
                    target_z=preplace_z,
                    motion_label=(
                        "Lower attached cube to safe "
                        "pre-place height"
                    ),
                    slowdown_factor=5.0,
                    position_tolerance=0.012,
                    skip_tolerance=0.002,
                )
            )

            if not descent_ok:
                return False

            # The cube must remain logically attached until
            # the explicit DETACH_CUBE state.
            if not self.scene_manager.verify_cube_attached():
                self.get_logger().error(
                    "The attached-cube state was lost during "
                    "the pre-place descent."
                )
                return False

            return True

        if description.state == TaskState.DETACH_CUBE:
            return self.scene_manager.detach_cube_from_hand()

        if description.state == TaskState.RELEASE_CUBE:
            open_position = float(
                self.get_parameter(
                    "open_gripper_position"
                ).value
            )

            self.get_logger().info(
                "Opening the gripper to physically release "
                "test_cube..."
            )

            release_ok = self.gripper_manager.command(
                position=open_position,
                max_effort=0.0,
                position_tolerance=0.003,
            )

            if not release_ok:
                self.get_logger().error(
                    "Physical cube release failed."
                )
                return False

            settle_time = float(
                self.get_parameter(
                    "release_settle_time"
                ).value
            )

            self.get_logger().info(
                "Gripper opened successfully. "
                "Waiting for the Gazebo cube to settle: "
                f"{settle_time:.1f} s"
            )

            time.sleep(settle_time)

            self.get_logger().info(
                "Release settling interval completed."
            )
            return True

        if description.state == TaskState.UPDATE_PLACED_SCENE:
            cube_z = float(
                self.get_parameter("cube_z").value
            )

            return (
                self.scene_manager
                .update_placed_cube_after_release(
                    placed_z=cube_z,
                )
            )

        if description.state == TaskState.RETREAT:
            cube_z = float(
                self.get_parameter("cube_z").value
            )

            if not self.scene_manager.verify_cube_placed(
                placed_z=cube_z,
            ):
                self.get_logger().error(
                    "Retreat blocked because the placed-cube "
                    "state is invalid."
                )
                return False

            retreat_distance = float(
                self.get_parameter(
                    "retreat_distance"
                ).value
            )

            current_transform = (
                self.motion_manager.lookup_tcp_transform()
            )
            current_position = (
                current_transform.transform.translation
            )

            target_z = (
                current_position.z + retreat_distance
            )

            self.get_logger().info(
                "Preparing vertical retreat after placement: "
                f"z={current_position.z:.3f} -> "
                f"{target_z:.3f} m"
            )

            retreat_ok = (
                self.motion_manager.move_cartesian_to(
                    target_x=current_position.x,
                    target_y=current_position.y,
                    target_z=target_z,
                    motion_label=(
                        "Vertical retreat after placement"
                    ),
                    slowdown_factor=5.0,
                    position_tolerance=0.012,
                    skip_tolerance=0.002,
                )
            )

            if not retreat_ok:
                return False

            # Confirm that the cube remained a placed world
            # object while the gripper moved away.
            return self.scene_manager.verify_cube_placed(
                placed_z=cube_z,
            )

        if description.state == TaskState.RETURN_HOME:
            cube_z = float(
                self.get_parameter("cube_z").value
            )

            if not self.scene_manager.verify_cube_placed(
                placed_z=cube_z,
            ):
                self.get_logger().error(
                    "Return Home blocked because the placed "
                    "cube state is invalid."
                )
                return False

            return (
                self.motion_manager
                .move_to_joint_state_ompl(
                    target_positions=HOME_JOINTS,
                    constraint_name=(
                        "collision_aware_return_home"
                    ),
                    velocity_scaling=0.10,
                    acceleration_scaling=0.10,
                )
            )

        if description.state == TaskState.COMPLETE:
            cube_z = float(
                self.get_parameter("cube_z").value
            )

            if not self.scene_manager.verify_cube_placed(
                placed_z=cube_z,
            ):
                self.get_logger().error(
                    "Final placed-cube verification failed."
                )
                return False

            if not self.motion_manager.verify_joint_positions(
                target_positions=HOME_JOINTS,
                tolerance=0.03,
            ):
                self.get_logger().error(
                    "Final Home-state verification failed."
                )
                return False

            self.get_logger().info(
                "AUTOMATED PICK-AND-PLACE COMPLETED "
                "SUCCESSFULLY."
            )
            self.get_logger().info(
                "Final state: Panda at Home, gripper open, "
                "test_cube placed, no attached object."
            )
            return True

        self.get_logger().error(
            f"Real execution for {description.label} "
            "has not been integrated yet."
        )
        return False

    def run(self) -> bool:
        self.print_task_parameters()

        print()
        print("=== PICK-AND-PLACE STATE SEQUENCE ===")

        total = len(TASK_SEQUENCE)

        for index, description in enumerate(
            TASK_SEQUENCE,
            start=1,
        ):
            print(
                f"[{index:02d}/{total:02d}] "
                f"{description.label}"
            )
            print(f"         {description.purpose}")

        print()

        if self.dry_run:
            print(
                "DRY RUN PASSED: state order and task "
                "parameters are valid."
            )
            print(
                "No Planning Scene, controller, or robot "
                "state was changed."
            )
            return True

        valid_labels = {
            item.label
            for item in TASK_SEQUENCE
        }

        if self.stop_after_state not in valid_labels:
            self.get_logger().error(
                "Invalid stop_after_state: "
                f"{self.stop_after_state}"
            )
            return False

        print()
        print("=== PARTIAL REAL EXECUTION ===")

        for description in TASK_SEQUENCE:
            print(
                f"[START] {description.label}: "
                f"{description.purpose}"
            )

            state_success = self.execute_state(
                description
            )

            if not state_success:
                print(
                    f"[FAIL] {description.label}"
                )
                print(
                    "TASK ABORTED: no dependent state "
                    "will be executed."
                )
                return False

            print(f"[PASS] {description.label}")

            if (
                description.label
                == self.stop_after_state
            ):
                print()
                print(
                    "PARTIAL REAL EXECUTION PASSED: "
                    f"stopped safely after "
                    f"{description.label}."
                )
                return True

        return True


def main(args=None) -> None:
    rclpy.init(args=args)

    checker = None
    state_machine = None
    success = False

    try:
        print("Running integrated preflight check...")

        checker = PreflightChecker()
        preflight_passed = checker.run_checks()

        checker.destroy_node()
        checker = None

        if not preflight_passed:
            print(
                "AUTOMATED TASK BLOCKED: "
                "preflight did not pass."
            )
            success = False
        else:
            state_machine = PickPlaceStateMachine()
            success = state_machine.run()

    except Exception as error:
        print(
            "AUTOMATED TASK ERROR: "
            f"{type(error).__name__}: {error}"
        )
        success = False

    finally:
        if checker is not None:
            checker.destroy_node()

        if state_machine is not None:
            state_machine.destroy_node()

        rclpy.shutdown()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
