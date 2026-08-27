#!/usr/bin/env python3

import sys

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    CollisionObject,
    PlanningScene,
    PlanningSceneComponents,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from shape_msgs.msg import SolidPrimitive


class PlacedCubeSceneUpdater(Node):
    def __init__(self) -> None:
        super().__init__("update_placed_cube_scene")

        self.apply_client = self.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
        )

        self.get_client = self.create_client(
            GetPlanningScene,
            "/get_planning_scene",
        )

    def apply_cube_pose(self) -> bool:
        if not self.apply_client.wait_for_service(
            timeout_sec=10.0
        ):
            self.get_logger().error(
                "/apply_planning_scene is unavailable."
            )
            return False

        cube = CollisionObject()
        cube.header.frame_id = "world"
        cube.id = "test_cube"

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [0.05, 0.05, 0.05]

        pose = Pose()
        pose.position.x = 0.55
        pose.position.y = 0.25
        pose.position.z = 0.425
        pose.orientation.w = 1.0

        # Store the object's world pose in CollisionObject.pose.
        cube.pose = pose

        # The box geometry is centred at the object's local origin.
        local_pose = Pose()
        local_pose.orientation.w = 1.0

        cube.primitives = [primitive]
        cube.primitive_poses = [local_pose]

        # ADD with an existing ID replaces that world object.
        cube.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.world.collision_objects = [cube]

        request = ApplyPlanningScene.Request()
        request.scene = scene

        self.get_logger().info(
            "Updating test_cube to placed pose: "
            "(0.55, 0.25, 0.425) m..."
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
                "MoveIt rejected the scene update."
            )
            return False

        return self.verify_scene()

    def verify_scene(self) -> bool:
        if not self.get_client.wait_for_service(
            timeout_sec=10.0
        ):
            self.get_logger().error(
                "/get_planning_scene is unavailable."
            )
            return False

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
            self.get_logger().error(
                "GetPlanningScene request timed out."
            )
            return False

        scene = future.result().scene

        attached_ids = [
            item.object.id
            for item in scene.robot_state.attached_collision_objects
        ]

        cube = next(
            (
                item
                for item in scene.world.collision_objects
                if item.id == "test_cube"
            ),
            None,
        )

        if cube is None:
            self.get_logger().error(
                "test_cube was not found in the MoveIt world."
            )
            return False

        if "test_cube" in attached_ids:
            self.get_logger().error(
                "test_cube is still marked as attached."
            )
            return False

        if not cube.primitive_poses:
            self.get_logger().error(
                "test_cube has no primitive pose."
            )
            return False

        object_position = cube.pose.position
        local_position = cube.primitive_poses[0].position

        self.get_logger().info(
            "MoveIt test_cube object pose in world: "
            f"x={object_position.x:.3f}, "
            f"y={object_position.y:.3f}, "
            f"z={object_position.z:.3f}"
        )

        self.get_logger().info(
            "Box geometry pose relative to object: "
            f"x={local_position.x:.3f}, "
            f"y={local_position.y:.3f}, "
            f"z={local_position.z:.3f}"
        )

        expected = (0.55, 0.25, 0.425)
        error = (
            abs(object_position.x - expected[0]),
            abs(object_position.y - expected[1]),
            abs(object_position.z - expected[2]),
        )

        if max(error) > 0.001:
            self.get_logger().error(
                "The MoveIt world pose does not match the "
                "requested placed pose."
            )
            return False

        self.get_logger().info(
            "Placed-cube Planning Scene synchronization succeeded."
        )
        return True


def main() -> None:
    rclpy.init()
    node = PlacedCubeSceneUpdater()

    try:
        success = node.apply_cube_pose()
    except Exception as error:
        node.get_logger().error(str(error))
        success = False
    finally:
        node.destroy_node()
        rclpy.shutdown()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
