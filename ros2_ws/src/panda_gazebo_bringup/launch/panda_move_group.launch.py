import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    gazebo_package_share = get_package_share_directory(
        "panda_gazebo_bringup"
    )

    moveit_package_share = get_package_share_directory(
        "moveit_resources_panda_moveit_config"
    )

    robot_urdf = os.path.join(
        gazebo_package_share,
        "urdf",
        "panda_arm_gazebo.urdf.xacro",
    )

    robot_srdf = os.path.join(
        gazebo_package_share,
        "config",
        "panda_arm_gazebo.srdf",
    )

    arm_controllers = os.path.join(
        moveit_package_share,
        "config",
        "moveit_controllers.yaml",
    )

    moveit_config = (
        MoveItConfigsBuilder(
            "moveit_resources_panda",
            package_name="moveit_resources_panda_moveit_config",
        )
        .robot_description(file_path=robot_urdf)
        .robot_description_semantic(file_path=robot_srdf)
        .trajectory_execution(file_path=arm_controllers)
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": True},
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    return LaunchDescription([move_group])
