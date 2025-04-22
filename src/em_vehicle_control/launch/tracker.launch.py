from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    robot_name = LaunchConfiguration('robot_name', default='robot1')

    tracker_node = Node(
        package='em_vehicle_control',
        executable='tracker',
        name='tracker',
        parameters=[{'robot_name': robot_name}],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('robot_name', default_value='robot1'),
        tracker_node
    ])
