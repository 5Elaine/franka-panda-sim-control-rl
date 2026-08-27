import os
import glob
from setuptools import find_packages, setup

package_name = 'panda_pick_place_demo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join(
                'share',
                package_name,
                'launch',
            ),
            glob.glob('launch/*.launch.py'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='5Elaine',
    maintainer_email='5Elaine@users.noreply.github.com',
    description='Automated Franka Panda pick-and-place state machine',
    license='Proprietary',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'preflight_checker = panda_pick_place_demo.preflight_checker:main',
            'pick_place_state_machine = panda_pick_place_demo.pick_place_state_machine:main',
        ],
    },
)
