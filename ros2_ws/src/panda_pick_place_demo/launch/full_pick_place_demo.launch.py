#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import (
    get_package_share_directory,
)

import os


def generate_launch_description():
    startup_delay = LaunchConfiguration(
        "startup_delay"
    )

    full_stack_launch = os.path.join(
        get_package_share_directory(
            "panda_gazebo_bringup"
        ),
        "launch",
        "panda_pick_full_stack.launch.py",
    )

    start_full_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            full_stack_launch
        )
    )

    start_state_machine = TimerAction(
        period=startup_delay,
        actions=[
            Node(
                package="panda_pick_place_demo",
                executable="pick_place_state_machine",
                name="automated_pick_place_task",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "use_sim_time": True,
                        "dry_run": False,
                        "stop_after_state": "COMPLETE",
                    }
                ],
            )
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "startup_delay",
                default_value="10.0",
                description=(
                    "Seconds to wait before starting the "
                    "automated task node."
                ),
            ),
            start_full_stack,
            start_state_machine,
        ]
    )
