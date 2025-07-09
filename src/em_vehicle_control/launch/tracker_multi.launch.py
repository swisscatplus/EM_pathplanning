from launch import LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import PushRosNamespace
from launch.substitutions import ThisLaunchFileDir, PathJoinSubstitution

def generate_launch_description():
    robots = ['robot1', 'robot2', 'robot3']
    launch_descriptions = []

    for robot in robots:
        group = GroupAction([
            PushRosNamespace(robot),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([ThisLaunchFileDir(), 'tracker.launch.py'])
                ),
                launch_arguments={'robot_name': robot}.items()
            )
        ])
        launch_descriptions.append(group)

    return LaunchDescription(launch_descriptions)
