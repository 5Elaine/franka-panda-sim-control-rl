#!/usr/bin/env python3

import sys

import rclpy
from rclpy.node import Node

from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    PlanningScene,
    PlanningSceneComponents,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene


class CubeAttacher(Node):
    def __init__(self) -> None:
        super().__init__("attach_cube_to_gripper")

        self.apply_client = self.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
        )

        self.get_client = self.create_client(
            GetPlanningScene,
            "/get_planning_scene",
        )

    def read_scene(self):
        if not self.get_client.wait_for_service(timeout_sec=10.0):
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
            self,
            future,
            timeout_sec=10.0,
        )

        if not future.done() or future.result() is None:
            raise RuntimeError(
                "GetPlanningScene request timed out."
            )

        return future.result().scene

    @staticmethod
    def object_ids(scene):
        world_ids = [
            item.id
            for item in scene.world.collision_objects
        ]

        attached_ids = [
            item.object.id
            for item
            in scene.robot_state.attached_collision_objects
        ]

        return world_ids, attached_ids

    def attach_existing_cube(self) -> bool:
        before = self.read_scene()
        world_before, attached_before = self.object_ids(before)

        self.get_logger().info(
            f"World objects before attachment: {world_before}"
        )
        self.get_logger().info(
            f"Attached objects before attachment: {attached_before}"
        )

        if "test_cube" in attached_before:
            self.get_logger().info(
                "test_cube is already attached."
            )
            return True

        if "test_cube" not in world_before:
            self.get_logger().error(
                "test_cube is not present in MoveIt world. "
                "Run apply_pick_scene.py first."
            )
            return False

        if not self.apply_client.wait_for_service(
            timeout_sec=10.0
        ):
            self.get_logger().error(
                "/apply_planning_scene is unavailable."
            )
            return False

        attached_cube = AttachedCollisionObject()

        # Attach to the physical hand link.
        attached_cube.link_name = "panda_hand"

        # Only reference the existing object by ID.
        # MoveIt transfers its existing geometry and pose.
        attached_cube.object.id = "test_cube"
        attached_cube.object.operation = CollisionObject.ADD

        # Collisions with these links are intentional during grasping.
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

        self.get_logger().info(
            "Attaching existing test_cube to panda_hand..."
        )

        future = self.apply_client.call_async(request)

        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=10.0,
        )

        if not future.done() or future.result() is None:
            self.get_logger().error(
                "ApplyPlanningScene request timed out."
            )
            return False

        if not future.result().success:
            self.get_logger().error(
                "MoveIt rejected the attachment."
            )
            return False

        after = self.read_scene()
        world_after, attached_after = self.object_ids(after)

        self.get_logger().info(
            f"World objects after attachment: {world_after}"
        )
        self.get_logger().info(
            f"Attached objects after attachment: {attached_after}"
        )

        valid = (
            "test_cube" not in world_after
            and "test_cube" in attached_after
            and "table" in world_after
        )

        if not valid:
            self.get_logger().error(
                "Attachment verification failed."
            )
            return False

        self.get_logger().info(
            "Planning Scene attachment verified successfully."
        )
        return True


def main() -> None:
    rclpy.init()
    node = CubeAttacher()

    try:
        success = node.attach_existing_cube()
    except Exception as error:
        node.get_logger().error(str(error))
        success = False
    finally:
        node.destroy_node()
        rclpy.shutdown()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
