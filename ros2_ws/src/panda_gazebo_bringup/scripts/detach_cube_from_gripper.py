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


class CubeDetacher(Node):
    def __init__(self) -> None:
        super().__init__("detach_cube_from_gripper")

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
            PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
            | PlanningSceneComponents.WORLD_OBJECT_NAMES
            | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
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
    def get_ids(scene):
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

    def detach_cube(self) -> bool:
        before = self.read_scene()
        world_before, attached_before = self.get_ids(before)

        self.get_logger().info(
            f"World objects before detach: {world_before}"
        )
        self.get_logger().info(
            f"Attached objects before detach: {attached_before}"
        )

        attached_cube = next(
            (
                item
                for item
                in before.robot_state.attached_collision_objects
                if item.object.id == "test_cube"
            ),
            None,
        )

        if attached_cube is None:
            if "test_cube" in world_before:
                self.get_logger().info(
                    "test_cube is already detached."
                )
                return True

            self.get_logger().error(
                "test_cube is not attached and is not in the world."
            )
            return False

        if not self.apply_client.wait_for_service(
            timeout_sec=10.0
        ):
            self.get_logger().error(
                "/apply_planning_scene is unavailable."
            )
            return False

        detach = AttachedCollisionObject()

        # Reuse the actual link instead of assuming which link
        # was used during attachment.
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

        self.get_logger().info(
            f"Detaching test_cube from {attached_cube.link_name}..."
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
                "MoveIt rejected the detach request."
            )
            return False

        after = self.read_scene()
        world_after, attached_after = self.get_ids(after)

        self.get_logger().info(
            f"World objects after detach: {world_after}"
        )
        self.get_logger().info(
            f"Attached objects after detach: {attached_after}"
        )

        valid = (
            "test_cube" in world_after
            and "test_cube" not in attached_after
            and "table" in world_after
        )

        if not valid:
            self.get_logger().error(
                "Detach verification failed."
            )
            return False

        self.get_logger().info(
            "Planning Scene detach verified successfully."
        )
        return True


def main() -> None:
    rclpy.init()
    node = CubeDetacher()

    try:
        success = node.detach_cube()
    except Exception as error:
        node.get_logger().error(str(error))
        success = False
    finally:
        node.destroy_node()
        rclpy.shutdown()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
