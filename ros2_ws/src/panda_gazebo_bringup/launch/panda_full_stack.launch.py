from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("panda_gazebo_bringup")

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [package_share, "launch", "panda_gazebo.launch.py"]
            )
        )
    )

    move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [package_share, "launch", "panda_move_group.launch.py"]
            )
        )
    )

    # Allow Gazebo and ros2_control to initialize before MoveIt starts.
    delayed_move_group = TimerAction(
        period=8.0,
        actions=[move_group_launch],
    )

    software_rendering = SetEnvironmentVariable(
        name="LIBGL_ALWAYS_SOFTWARE",
        value="1",
    )

    return LaunchDescription(
        [
            software_rendering,
            gazebo_launch,
            delayed_move_group,
        ]
    )
