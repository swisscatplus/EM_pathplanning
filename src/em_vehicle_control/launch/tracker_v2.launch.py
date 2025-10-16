# launch/multi_robot.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace
from launch.actions import GroupAction
import os
import yaml

# Change this if your real package name differs (must match setup.py name)
PKG = "em_vehicle_control"

def _pkg_dir():
    # .../em_vehicle_control/launch -> parent is the package root
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def _require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    return path

def generate_launch_description():
    pkg_dir = _pkg_dir()
    config_dir = os.path.join(pkg_dir, "config")

    # Hardcoded config files living next to launch/ (sibling dir)
    roads_yaml  = _require(os.path.join(config_dir, "roads.yaml"))
    robots_yaml = _require(os.path.join(config_dir, "robots.yaml"))

    with open(robots_yaml, "r") as f:
        robots_cfg = yaml.safe_load(f) or {}

    frame_id = robots_cfg.get("frame_id", "map")
    robots = robots_cfg.get("robots", [])
    if not robots:
        raise RuntimeError("robots.yaml has no 'robots' entries.")

    # --- Single planner (global namespace) ---
    planner = Node(
        package=PKG,
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
            package=PKG,
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
