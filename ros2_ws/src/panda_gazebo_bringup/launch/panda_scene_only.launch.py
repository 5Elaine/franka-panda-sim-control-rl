from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("panda_gazebo_bringup")

    world_file = PathJoinSubstitution(
        [
            package_share,
            "worlds",
            "panda_pick_scene.sdf",
        ]
    )

    software_rendering = SetEnvironmentVariable(
        name="LIBGL_ALWAYS_SOFTWARE",
        value="1",
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                FindPackageShare("ros_gz_sim"),
                "/launch/gz_sim.launch.py",
            ]
        ),
        launch_arguments={
            # Do not use -r: start paused so the cube fall can be observed.
            "gz_args": ["-v 3 ", world_file],
        }.items(),
    )

    return LaunchDescription(
        [
            software_rendering,
            gazebo,
        ]
    )
