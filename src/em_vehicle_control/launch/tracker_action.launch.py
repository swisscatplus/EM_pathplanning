from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    tracker_node = Node(
        package='em_vehicle_control',
        executable='tracker',
        name='tracker',
        output='screen',
        parameters=[{
            'robot_name': 'robot1',
            'map_frame': 'map',
            'base_link_frame': 'base_link',
        }],
        arguments=['--ros-args', '--log-level', 'info'],
    )

    return LaunchDescription([
        tracker_node,
    ])
