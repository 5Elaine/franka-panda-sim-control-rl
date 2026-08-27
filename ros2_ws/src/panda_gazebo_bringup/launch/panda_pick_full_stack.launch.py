from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("panda_gazebo_bringup")

    gazebo_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    package_share,
                    "launch",
                    "panda_pick_gazebo.launch.py",
                ]
            )
        )
    )

    moveit_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    package_share,
                    "launch",
                    "panda_hand_move_group.launch.py",
                ]
            )
        )
    )

    delayed_moveit = TimerAction(
        period=10.0,
        actions=[moveit_stack],
    )

    return LaunchDescription(
        [
            gazebo_stack,
            delayed_moveit,
        ]
    )
