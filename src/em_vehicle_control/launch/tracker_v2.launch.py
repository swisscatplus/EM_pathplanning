from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace
from launch.actions import GroupAction
from ament_index_python.packages import get_package_share_directory
import os
import yaml

pkg_name = 'em_vehicle_control'

roads_yaml = os.path.join(
    get_package_share_directory(pkg_name),
    'config',
    'road.yaml'
)
robots_yaml = os.path.join(
    get_package_share_directory(pkg_name),
    'config',
    'robots.yaml'
)

def generate_launch_description():

    with open(robots_yaml, "r") as f:
        robots_cfg = yaml.safe_load(f) or {}

    frame_id = robots_cfg.get("frame_id", "map")
    robots = robots_cfg.get("robots", [])
    if not robots:
        raise RuntimeError("robots.yaml has no 'robots' entries.")

    # --- Single planner (global namespace) ---
    planner = Node(
        package=pkg_name,
        executable="planner_v2",
        name="planner",
        output="screen",
        parameters=[{
            # Hardcoded absolute path; not a user param
            "map_config": roads_yaml,
            "frame_id": frame_id,          # optional but handy to centralize
            # "timer_period_s": robots_cfg.get("timer_period_s", 2.0),
        }],
    )

    # --- One tracker per robot, under its namespace ---
    tracker_groups = []
    for r in robots:
        name = r.get("name", "")
        if not name:
            raise RuntimeError("Each robot in robots.yaml must have a 'name'.")
        ns = r.get("namespace", name)
        base_link_frame = r.get("base_link_frame", f"{name}/base_link")

        tracker = Node(
            package=pkg_name,
            executable="tracker_v2",
            name="tracker",
            output="screen",
            parameters=[{
                "robot_name": name,
                "map_frame": frame_id,
                "base_link_frame": base_link_frame,
            }],
        )

        tracker_groups.append(GroupAction([
            PushRosNamespace(ns),
            tracker,
        ]))

    return LaunchDescription([planner, *tracker_groups])
