#!/usr/bin/env python3

import sys

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive


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


class PickSceneSetup(Node):
    def __init__(self) -> None:
        super().__init__("pick_scene_setup")

        self.client = self.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
        )

    def apply_scene(self) -> bool:
        self.get_logger().info(
            "Waiting for /apply_planning_scene..."
        )

        if not self.client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                "/apply_planning_scene is unavailable."
            )
            return False

        table = make_box(
            object_id="table",
            size_xyz=(0.80, 0.80, 0.40),
            position_xyz=(0.55, 0.0, 0.20),
        )

        cube = make_box(
            object_id="test_cube",
            size_xyz=(0.05, 0.05, 0.05),
            position_xyz=(0.55, 0.0, 0.425),
        )

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.world.collision_objects = [table, cube]

        request = ApplyPlanningScene.Request()
        request.scene = scene

        self.get_logger().info(
            "Adding table and test_cube to MoveIt Planning Scene..."
        )

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=10.0,
        )

        if not future.done():
            self.get_logger().error(
                "ApplyPlanningScene request timed out."
            )
            return False

        response = future.result()

        if response is None or not response.success:
            self.get_logger().error(
                "MoveIt rejected the planning-scene update."
            )
            return False

        self.get_logger().info(
            "Planning Scene updated successfully."
        )
        return True


def main() -> None:
    rclpy.init()

    node = PickSceneSetup()

    try:
        success = node.apply_scene()
    finally:
        node.destroy_node()
        rclpy.shutdown()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
