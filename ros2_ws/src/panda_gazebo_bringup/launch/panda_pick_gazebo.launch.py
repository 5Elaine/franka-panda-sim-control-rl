from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
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

    robot_xacro = PathJoinSubstitution(
        [
            package_share,
            "urdf",
            "panda_arm_hand_gazebo.urdf.xacro",
        ]
    )

    robot_description = ParameterValue(
        Command(["xacro ", robot_xacro]),
        value_type=str,
    )

    # VMware 中固定使用软件渲染，避免 Ogre2 视口闪烁或黑线。
    software_rendering = SetEnvironmentVariable(
        name="LIBGL_ALWAYS_SOFTWARE",
        value="1",
    )

    # 让 Gazebo 能找到 franka_description 中的网格资源。
    set_ign_resource_path = SetEnvironmentVariable(
        name="IGN_GAZEBO_RESOURCE_PATH",
        value="/opt/ros/humble/share",
    )

    set_gz_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value="/opt/ros/humble/share",
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                FindPackageShare("ros_gz_sim"),
                "/launch/gz_sim.launch.py",
            ]
        ),
        launch_arguments={
            "gz_args": ["-r -v 3 ", world_file],
        }.items(),
    )

    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
        ],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": True,
            }
        ],
    )

    spawn_panda = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-world",
            "panda_pick_world",
            "-name",
            "panda",
            "-topic",
            "robot_description",
            "-z",
            "0.0",
        ],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "60",
        ],
    )

    panda_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            "panda_arm_controller",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "60",
        ],
    )

    panda_hand_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            "panda_hand_controller",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "60",
        ],
    )

    # 机器人生成后，再加载三套控制器。
    start_controllers_after_spawn = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_panda,
            on_exit=[
                joint_state_broadcaster_spawner,
                panda_arm_controller_spawner,
                panda_hand_controller_spawner,
            ],
        )
    )

    return LaunchDescription(
        [
            software_rendering,
            set_ign_resource_path,
            set_gz_resource_path,
            gazebo,
            clock_bridge,
            robot_state_publisher,
            spawn_panda,
            start_controllers_after_spawn,
        ]
    )
