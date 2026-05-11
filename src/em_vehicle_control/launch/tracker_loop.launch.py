from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    robot_name = LaunchConfiguration('robot_name', default='robot1')
    paths_config = LaunchConfiguration('paths_config', default='paths.yaml')
    workflow_config = LaunchConfiguration('workflow_config', default='path_workflows.yaml')
    workflow = LaunchConfiguration('workflow', default='full_loop')

    tracker_node = Node(
        package='em_vehicle_control',
        executable='tracker',
        name='tracker',
        parameters=[{
            'robot_name': robot_name,
            'paths_config': paths_config,
        }],
        output='screen',
        arguments=['--ros-args', '--log-level', 'info'],
    )

    path_node = Node(
        package='em_vehicle_control',
        executable='publish_path_loop',
        name='path_publisher_loop',
        output='screen',
        parameters=[{
            'paths_config': paths_config,
            'workflow_config': workflow_config,
            'workflow': workflow,
            'start_immediately': True,
        }],
        arguments=['--ros-args', '--log-level', 'info'],
    )

    return LaunchDescription([
        DeclareLaunchArgument('paths_config', default_value='paths.yaml'),
        DeclareLaunchArgument('workflow_config', default_value='path_workflows.yaml'),
        DeclareLaunchArgument('workflow', default_value='full_loop'),
        tracker_node,
        path_node
    ])
