#!/usr/bin/env python3

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    PlanningScene,
    PlanningSceneComponents,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive


class SceneManager:
    """Manage and verify MoveIt Planning Scene objects."""

    TABLE_SIZE = (0.80, 0.80, 0.40)
    TABLE_POSITION = (0.55, 0.00, 0.20)

    CUBE_SIZE = (0.05, 0.05, 0.05)
    CUBE_POSITION = (0.55, 0.00, 0.425)

    def __init__(self, node: Node) -> None:
        self.node = node

        self.apply_client = node.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
        )

        self.get_client = node.create_client(
            GetPlanningScene,
            "/get_planning_scene",
        )

    @staticmethod
    def make_box(
        object_id: str,
        size_xyz: tuple[float, float, float],
        position_xyz: tuple[float, float, float],
    ) -> CollisionObject:
        collision_object = CollisionObject()
        collision_object.header.frame_id = "world"
        collision_object.id = object_id

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = list(size_xyz)

        pose = Pose()
        pose.position.x = position_xyz[0]
        pose.position.y = position_xyz[1]
        pose.position.z = position_xyz[2]
        pose.orientation.w = 1.0

        collision_object.primitives = [box]
        collision_object.primitive_poses = [pose]
        collision_object.operation = CollisionObject.ADD

        return collision_object

    def reset_attached_cube_if_needed(self) -> bool:
        """
        Remove a stale test_cube attachment left by an earlier
        partial run before restoring the initial scene.
        """
        try:
            current_scene = self.read_scene()
        except Exception as error:
            self.node.get_logger().error(
                f"Unable to inspect current scene: {error}"
            )
            return False

        attached_cube = next(
            (
                item
                for item
                in current_scene.robot_state.attached_collision_objects
                if item.object.id == "test_cube"
            ),
            None,
        )

        if attached_cube is None:
            self.node.get_logger().info(
                "No stale test_cube attachment found."
            )
            return True

        if not self.apply_client.wait_for_service(
            timeout_sec=10.0
        ):
            self.node.get_logger().error(
                "/apply_planning_scene is unavailable."
            )
            return False

        detach = AttachedCollisionObject()
        detach.link_name = attached_cube.link_name
        detach.object.id = "test_cube"
        detach.object.operation = CollisionObject.REMOVE

        scene_diff = PlanningScene()
        scene_diff.is_diff = True
        scene_diff.robot_state.is_diff = True
        scene_diff.robot_state.attached_collision_objects = [
            detach
        ]

        request = ApplyPlanningScene.Request()
        request.scene = scene_diff

        self.node.get_logger().info(
            "Removing stale test_cube attachment from "
            f"{attached_cube.link_name}..."
        )

        future = self.apply_client.call_async(request)

        rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=10.0,
        )

        if (
            not future.done()
            or future.result() is None
        ):
            self.node.get_logger().error(
                "Stale-attachment removal timed out."
            )
            return False

        if not future.result().success:
            self.node.get_logger().error(
                "MoveIt rejected stale-attachment removal."
            )
            return False

        self.node.get_logger().info(
            "Stale test_cube attachment removed."
        )
        return True

    def initialize_pick_scene(self) -> bool:
        if not self.reset_attached_cube_if_needed():
            return False

        if not self.apply_client.wait_for_service(
            timeout_sec=10.0
        ):
            self.node.get_logger().error(
                "/apply_planning_scene is unavailable."
            )
            return False

        table = self.make_box(
            object_id="table",
            size_xyz=self.TABLE_SIZE,
            position_xyz=self.TABLE_POSITION,
        )

        cube = self.make_box(
            object_id="test_cube",
            size_xyz=self.CUBE_SIZE,
            position_xyz=self.CUBE_POSITION,
        )

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.world.collision_objects = [table, cube]

        request = ApplyPlanningScene.Request()
        request.scene = scene

        self.node.get_logger().info(
            "Applying initial table and test_cube "
            "to the MoveIt Planning Scene..."
        )

        future = self.apply_client.call_async(request)

        rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=10.0,
        )

        if not future.done() or future.result() is None:
            self.node.get_logger().error(
                "ApplyPlanningScene request timed out."
            )
            return False

        if not future.result().success:
            self.node.get_logger().error(
                "MoveIt rejected the initial scene."
            )
            return False

        return self.verify_initial_scene()

    def attach_cube_to_hand(self) -> bool:
        """
        Transfer the existing test_cube world object into the
        robot's attached-object list.
        """
        try:
            before = self.read_scene()
        except Exception as error:
            self.node.get_logger().error(str(error))
            return False

        world_before = {
            item.id
            for item in before.world.collision_objects
        }

        attached_before = {
            item.object.id
            for item
            in before.robot_state.attached_collision_objects
        }

        self.node.get_logger().info(
            "World objects before attachment: "
            f"{sorted(world_before)}"
        )
        self.node.get_logger().info(
            "Attached objects before attachment: "
            f"{sorted(attached_before)}"
        )

        if "test_cube" in attached_before:
            self.node.get_logger().info(
                "test_cube is already attached."
            )
            return True

        if "test_cube" not in world_before:
            self.node.get_logger().error(
                "test_cube is not present in the MoveIt world."
            )
            return False

        if not self.apply_client.wait_for_service(
            timeout_sec=10.0
        ):
            self.node.get_logger().error(
                "/apply_planning_scene is unavailable."
            )
            return False

        attached_cube = AttachedCollisionObject()
        attached_cube.link_name = "panda_hand"

        # Reference the existing MoveIt world object by ID.
        attached_cube.object.id = "test_cube"
        attached_cube.object.operation = CollisionObject.ADD

        # Contact with the gripper links is expected.
        attached_cube.touch_links = [
            "panda_hand",
            "panda_hand_tcp",
            "panda_leftfinger",
            "panda_rightfinger",
        ]

        scene_diff = PlanningScene()
        scene_diff.is_diff = True
        scene_diff.robot_state.is_diff = True
        scene_diff.robot_state.attached_collision_objects = [
            attached_cube
        ]

        request = ApplyPlanningScene.Request()
        request.scene = scene_diff

        self.node.get_logger().info(
            "Attaching existing test_cube to panda_hand..."
        )

        future = self.apply_client.call_async(request)

        rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=10.0,
        )

        if (
            not future.done()
            or future.result() is None
        ):
            self.node.get_logger().error(
                "Attachment request timed out."
            )
            return False

        if not future.result().success:
            self.node.get_logger().error(
                "MoveIt rejected the attachment."
            )
            return False

        try:
            after = self.read_scene()
        except Exception as error:
            self.node.get_logger().error(str(error))
            return False

        world_after = {
            item.id
            for item in after.world.collision_objects
        }

        attached_objects = {
            item.object.id: item
            for item
            in after.robot_state.attached_collision_objects
        }

        self.node.get_logger().info(
            "World objects after attachment: "
            f"{sorted(world_after)}"
        )
        self.node.get_logger().info(
            "Attached objects after attachment: "
            f"{sorted(attached_objects)}"
        )

        attached_result = attached_objects.get(
            "test_cube"
        )

        valid = (
            "test_cube" not in world_after
            and attached_result is not None
            and attached_result.link_name == "panda_hand"
            and "table" in world_after
        )

        if not valid:
            self.node.get_logger().error(
                "Planning Scene attachment verification failed."
            )
            return False

        self.node.get_logger().info(
            "test_cube attachment verified successfully."
        )
        return True

    def detach_cube_from_hand(self) -> bool:
        """
        Remove test_cube from the robot's attached-object list
        and restore it as a MoveIt world object.
        """
        try:
            before = self.read_scene()
        except Exception as error:
            self.node.get_logger().error(str(error))
            return False

        world_before = {
            item.id
            for item in before.world.collision_objects
        }

        attached_cube = next(
            (
                item
                for item
                in before.robot_state.attached_collision_objects
                if item.object.id == "test_cube"
            ),
            None,
        )

        self.node.get_logger().info(
            "World objects before detach: "
            f"{sorted(world_before)}"
        )

        attached_before_ids = [
            item.object.id
            for item
            in before.robot_state.attached_collision_objects
        ]

        self.node.get_logger().info(
            "Attached objects before detach: "
            f"{sorted(attached_before_ids)}"
        )

        if attached_cube is None:
            if "test_cube" in world_before:
                self.node.get_logger().info(
                    "test_cube is already detached."
                )
                return True

            self.node.get_logger().error(
                "test_cube is neither attached nor present "
                "in the MoveIt world."
            )
            return False

        if not self.apply_client.wait_for_service(
            timeout_sec=10.0
        ):
            self.node.get_logger().error(
                "/apply_planning_scene is unavailable."
            )
            return False

        detach = AttachedCollisionObject()
        detach.link_name = attached_cube.link_name
        detach.object.id = "test_cube"
        detach.object.operation = CollisionObject.REMOVE

        scene_diff = PlanningScene()
        scene_diff.is_diff = True
        scene_diff.robot_state.is_diff = True
        scene_diff.robot_state.attached_collision_objects = [
            detach
        ]

        request = ApplyPlanningScene.Request()
        request.scene = scene_diff

        self.node.get_logger().info(
            "Detaching test_cube from "
            f"{attached_cube.link_name}..."
        )

        future = self.apply_client.call_async(request)

        rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=10.0,
        )

        if (
            not future.done()
            or future.result() is None
        ):
            self.node.get_logger().error(
                "Detach request timed out."
            )
            return False

        if not future.result().success:
            self.node.get_logger().error(
                "MoveIt rejected the detach request."
            )
            return False

        try:
            after = self.read_scene()
        except Exception as error:
            self.node.get_logger().error(str(error))
            return False

        world_after = {
            item.id: item
            for item in after.world.collision_objects
        }

        attached_after = {
            item.object.id
            for item
            in after.robot_state.attached_collision_objects
        }

        self.node.get_logger().info(
            "World objects after detach: "
            f"{sorted(world_after)}"
        )

        self.node.get_logger().info(
            "Attached objects after detach: "
            f"{sorted(attached_after)}"
        )

        valid = (
            "table" in world_after
            and "test_cube" in world_after
            and "test_cube" not in attached_after
        )

        if not valid:
            self.node.get_logger().error(
                "Planning Scene detach verification failed."
            )
            return False

        cube_position = self.effective_box_position(
            world_after["test_cube"]
        )

        self.node.get_logger().info(
            "Detached test_cube MoveIt position: "
            f"x={cube_position[0]:.3f}, "
            f"y={cube_position[1]:.3f}, "
            f"z={cube_position[2]:.3f}"
        )

        self.node.get_logger().info(
            "test_cube detach verified successfully."
        )
        return True

    def update_placed_cube_after_release(
        self,
        placed_z: float,
    ) -> bool:
        """
        Update the detached MoveIt world object after the
        physical Gazebo cube has fallen onto the table.

        Preserve the detached object's measured x and y, while
        replacing its suspended z with the table-resting z.
        """
        try:
            before = self.read_scene()
        except Exception as error:
            self.node.get_logger().error(str(error))
            return False

        world_before = {
            item.id: item
            for item in before.world.collision_objects
        }

        attached_before = {
            item.object.id
            for item
            in before.robot_state.attached_collision_objects
        }

        if "test_cube" in attached_before:
            self.node.get_logger().error(
                "Cannot update the placed cube while it is "
                "still marked as attached."
            )
            return False

        cube_before = world_before.get("test_cube")

        if cube_before is None:
            self.node.get_logger().error(
                "test_cube is missing from the MoveIt world."
            )
            return False

        if "table" not in world_before:
            self.node.get_logger().error(
                "table is missing from the MoveIt world."
            )
            return False

        current_position = self.effective_box_position(
            cube_before
        )

        placed_position = (
            current_position[0],
            current_position[1],
            placed_z,
        )

        self.node.get_logger().info(
            "Updating released test_cube position: "
            f"({current_position[0]:.3f}, "
            f"{current_position[1]:.3f}, "
            f"{current_position[2]:.3f}) -> "
            f"({placed_position[0]:.3f}, "
            f"{placed_position[1]:.3f}, "
            f"{placed_position[2]:.3f}) m"
        )

        if not self.apply_client.wait_for_service(
            timeout_sec=10.0
        ):
            self.node.get_logger().error(
                "/apply_planning_scene is unavailable."
            )
            return False

        placed_cube = self.make_box(
            object_id="test_cube",
            size_xyz=self.CUBE_SIZE,
            position_xyz=placed_position,
        )

        scene_diff = PlanningScene()
        scene_diff.is_diff = True
        scene_diff.robot_state.is_diff = True
        scene_diff.world.collision_objects = [
            placed_cube
        ]

        request = ApplyPlanningScene.Request()
        request.scene = scene_diff

        future = self.apply_client.call_async(request)

        rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=10.0,
        )

        if (
            not future.done()
            or future.result() is None
        ):
            self.node.get_logger().error(
                "Placed-cube scene update timed out."
            )
            return False

        if not future.result().success:
            self.node.get_logger().error(
                "MoveIt rejected the placed-cube update."
            )
            return False

        try:
            after = self.read_scene()
        except Exception as error:
            self.node.get_logger().error(str(error))
            return False

        world_after = {
            item.id: item
            for item in after.world.collision_objects
        }

        attached_after = {
            item.object.id
            for item
            in after.robot_state.attached_collision_objects
        }

        cube_after = world_after.get("test_cube")

        if cube_after is None:
            self.node.get_logger().error(
                "test_cube disappeared after the scene update."
            )
            return False

        if "test_cube" in attached_after:
            self.node.get_logger().error(
                "test_cube unexpectedly became attached again."
            )
            return False

        if "table" not in world_after:
            self.node.get_logger().error(
                "table disappeared after the scene update."
            )
            return False

        actual_position = self.effective_box_position(
            cube_after
        )

        self.node.get_logger().info(
            "Verified placed test_cube position: "
            f"x={actual_position[0]:.3f}, "
            f"y={actual_position[1]:.3f}, "
            f"z={actual_position[2]:.3f}"
        )

        if not self.values_close(
            actual_position,
            placed_position,
            tolerance=0.002,
        ):
            self.node.get_logger().error(
                "Placed-cube position verification failed. "
                f"Expected {placed_position}, "
                f"got {actual_position}."
            )
            return False

        self.node.get_logger().info(
            "Placed-cube Planning Scene synchronization "
            "succeeded."
        )
        return True

    def verify_cube_placed(
        self,
        placed_z: float,
        tolerance: float = 0.003,
    ) -> bool:
        """Verify that test_cube is a placed world object."""
        try:
            scene = self.read_scene()
        except Exception as error:
            self.node.get_logger().error(
                f"Unable to verify placed cube: {error}"
            )
            return False

        world_objects = {
            item.id: item
            for item in scene.world.collision_objects
        }

        attached_ids = {
            item.object.id
            for item
            in scene.robot_state.attached_collision_objects
        }

        cube = world_objects.get("test_cube")

        if cube is None:
            self.node.get_logger().error(
                "test_cube is missing from the MoveIt world."
            )
            return False

        if "test_cube" in attached_ids:
            self.node.get_logger().error(
                "test_cube is still marked as attached."
            )
            return False

        if "table" not in world_objects:
            self.node.get_logger().error(
                "table is missing from the MoveIt world."
            )
            return False

        position = self.effective_box_position(cube)

        self.node.get_logger().info(
            "Placed-cube state: "
            f"x={position[0]:.3f}, "
            f"y={position[1]:.3f}, "
            f"z={position[2]:.3f}"
        )

        if abs(position[2] - placed_z) > tolerance:
            self.node.get_logger().error(
                "Placed-cube height verification failed: "
                f"expected z={placed_z:.3f}, "
                f"got z={position[2]:.3f}."
            )
            return False

        self.node.get_logger().info(
            "Placed-cube state verified successfully."
        )
        return True

    def verify_cube_attached(
        self,
        expected_link: str = "panda_hand",
    ) -> bool:
        """Verify that test_cube is attached to the expected link."""
        try:
            scene = self.read_scene()
        except Exception as error:
            self.node.get_logger().error(
                f"Unable to verify attachment: {error}"
            )
            return False

        world_ids = {
            item.id
            for item in scene.world.collision_objects
        }

        attached_cube = next(
            (
                item
                for item
                in scene.robot_state.attached_collision_objects
                if item.object.id == "test_cube"
            ),
            None,
        )

        if attached_cube is None:
            self.node.get_logger().error(
                "test_cube is not present in the "
                "attached-object list."
            )
            return False

        if "test_cube" in world_ids:
            self.node.get_logger().error(
                "test_cube is simultaneously present in "
                "the MoveIt world and attached-object list."
            )
            return False

        if attached_cube.link_name != expected_link:
            self.node.get_logger().error(
                "test_cube is attached to an unexpected link: "
                f"{attached_cube.link_name}"
            )
            return False

        if "table" not in world_ids:
            self.node.get_logger().error(
                "table is missing from the MoveIt world."
            )
            return False

        self.node.get_logger().info(
            "Attached-cube state verified: "
            f"test_cube -> {attached_cube.link_name}"
        )
        return True

    def read_scene(self):
        if not self.get_client.wait_for_service(
            timeout_sec=10.0
        ):
            raise RuntimeError(
                "/get_planning_scene is unavailable."
            )

        request = GetPlanningScene.Request()
        request.components.components = (
            PlanningSceneComponents.WORLD_OBJECT_NAMES
            | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
            | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        )

        future = self.get_client.call_async(request)

        rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=10.0,
        )

        if not future.done() or future.result() is None:
            raise RuntimeError(
                "GetPlanningScene request timed out."
            )

        return future.result().scene

    @staticmethod
    def effective_box_position(
        collision_object: CollisionObject,
    ) -> tuple[float, float, float]:
        """
        Return the effective world position for the current
        axis-aligned boxes.

        MoveIt may store the world displacement in object.pose,
        primitive_poses[0], or split it between both.
        """
        object_position = collision_object.pose.position

        if collision_object.primitive_poses:
            primitive_position = (
                collision_object.primitive_poses[0].position
            )
        else:
            primitive_position = Pose().position

        return (
            object_position.x + primitive_position.x,
            object_position.y + primitive_position.y,
            object_position.z + primitive_position.z,
        )

    @staticmethod
    def values_close(
        actual,
        expected,
        tolerance: float = 0.001,
    ) -> bool:
        return all(
            abs(a - e) <= tolerance
            for a, e in zip(actual, expected)
        )

    def verify_box(
        self,
        collision_object: CollisionObject,
        expected_size,
        expected_position,
    ) -> bool:
        if not collision_object.primitives:
            self.node.get_logger().error(
                f"{collision_object.id} has no primitive geometry."
            )
            return False

        primitive = collision_object.primitives[0]

        if primitive.type != SolidPrimitive.BOX:
            self.node.get_logger().error(
                f"{collision_object.id} is not a BOX."
            )
            return False

        actual_size = tuple(primitive.dimensions)
        actual_position = self.effective_box_position(
            collision_object
        )

        self.node.get_logger().info(
            f"{collision_object.id}: "
            f"size={tuple(round(v, 3) for v in actual_size)}, "
            f"position="
            f"{tuple(round(v, 3) for v in actual_position)}"
        )

        size_valid = self.values_close(
            actual_size,
            expected_size,
        )

        position_valid = self.values_close(
            actual_position,
            expected_position,
        )

        if not size_valid:
            self.node.get_logger().error(
                f"{collision_object.id} size mismatch: "
                f"expected {expected_size}, got {actual_size}"
            )

        if not position_valid:
            self.node.get_logger().error(
                f"{collision_object.id} position mismatch: "
                f"expected {expected_position}, "
                f"got {actual_position}"
            )

        return size_valid and position_valid

    def verify_initial_scene(self) -> bool:
        try:
            scene = self.read_scene()
        except Exception as error:
            self.node.get_logger().error(str(error))
            return False

        world_objects = {
            item.id: item
            for item in scene.world.collision_objects
        }

        attached_ids = {
            item.object.id
            for item
            in scene.robot_state.attached_collision_objects
        }

        self.node.get_logger().info(
            f"MoveIt world objects: "
            f"{sorted(world_objects.keys())}"
        )

        self.node.get_logger().info(
            f"MoveIt attached objects: "
            f"{sorted(attached_ids)}"
        )

        missing = {
            "table",
            "test_cube",
        } - set(world_objects)

        if missing:
            self.node.get_logger().error(
                "Initial scene is missing: "
                + ", ".join(sorted(missing))
            )
            return False

        if "test_cube" in attached_ids:
            self.node.get_logger().error(
                "test_cube should initially be a world object, "
                "not an attached object."
            )
            return False

        table_valid = self.verify_box(
            world_objects["table"],
            self.TABLE_SIZE,
            self.TABLE_POSITION,
        )

        cube_valid = self.verify_box(
            world_objects["test_cube"],
            self.CUBE_SIZE,
            self.CUBE_POSITION,
        )

        if not table_valid or not cube_valid:
            return False

        self.node.get_logger().info(
            "Initial Planning Scene verified successfully."
        )

        return True
