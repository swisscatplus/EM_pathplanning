from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'em_vehicle_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, f'{package_name}.helper_classes'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planner = em_vehicle_control.planner:main',
            'tracker = em_vehicle_control.tracker_mpc:main',
            'planner_demo = em_vehicle_control.planner_demo:main',
            'tracker_demo = em_vehicle_control.tracker_mpc_demo:main',
            'publish_path = em_vehicle_control.publish_path:main',
            # 'pose2d_publisher = em_vehicle_control.pose2d_publisher:main',
            # 'pose2d_subscriber = em_vehicle_control.pose2d_subscriber:main',
        ],
    },
)
