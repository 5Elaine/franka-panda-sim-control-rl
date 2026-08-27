from setuptools import find_packages, setup

package_name = 'panda_motion_demo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='5Elaine',
    maintainer_email='5Elaine@users.noreply.github.com',
    description='ROS 2 motion and IK examples for the Franka Panda',
    license='Proprietary',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'simple_node = panda_motion_demo.simple_node:main',
            'joint_trajectory_client = panda_motion_demo.joint_trajectory_client:main',
            'move_to_home = panda_motion_demo.move_to_home:main',
            'move_target_then_home = panda_motion_demo.move_target_then_home:main',
            'pick_motion_sequence = panda_motion_demo.pick_motion_sequence:main',
            'gripper_control = panda_motion_demo.gripper_control:main',
            'pick_demo = panda_motion_demo.pick_demo:main',
            'end_effector_pose_listener = panda_motion_demo.end_effector_pose_listener:main',
            'compute_ik_from_current_pose = panda_motion_demo.compute_ik_from_current_pose:main',
            'move_to_ik_lifted_pose = panda_motion_demo.move_to_ik_lifted_pose:main',
            'ik_lift_and_execute = panda_motion_demo.ik_lift_and_execute:main',
        ],
    },
)
