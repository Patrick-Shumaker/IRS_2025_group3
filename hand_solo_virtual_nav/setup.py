from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'hand_solo_virtual_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/rviz',   glob('rviz/*.rviz')),
        ('share/' + package_name + '/map', glob('map/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maleen',
    maintainer_email='maleen.nj@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['hs_waypoint_follower = hand_solo_virtual_nav.hs_waypoint_follower:main',
        ],
    },
)
