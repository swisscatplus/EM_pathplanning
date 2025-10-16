from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import yaml
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

import tf2_ros
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header, ColorRGBA
from builtin_interfaces.msg import Time as TimeMsg
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point as RosPoint

from shapely.geometry import Polygon
from shapely.ops import triangulate

from ament_index_python.packages import get_package_share_directory

from em_vehicle_control_msgs.msg import Pose2D, Path2D
from em_vehicle_control_msgs.srv import PlanPath, CancelPath  # <-- new services
from em_vehicle_control.helper_classes.map import RoadSegment, RoadMap, RoadTrack
from em_vehicle_control.helper_classes.RNBTree import FleetManagerTree
from em_vehicle_control.helper_classes.vehicles import EdyMobile  # extend if needed

PoseYaw = Tuple[float, float, float]  # (x, y, yaw)


class Planner(Node):
    def __init__(self) -> None:
        super().__init__("planner")

        # ---------------- Parameters ----------------
        self.declare_parameter("map_config", "")      # YAML with 'frame_id', 'timer_period_s', 'roads'
        # Defaults used only if not in YAML:
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("timer_period_s", 2.0)

        # Load YAML (roads + frame/timer if provided)
        cfg_path = self._resolve_yaml_path(self.get_parameter("map_config").value)
        config = self._load_yaml(cfg_path)

        self.frame_id: str = self._get_param_or_yaml("frame_id", config, default="map")
        self.timer_period: float = float(self._get_param_or_yaml("timer_period_s", config, default=2.0))

        self.get_logger().info(f"Using map: {cfg_path}")
        self.get_logger().info(f"frame_id='{self.frame_id}', timer_period={self.timer_period}s")

        # ---------------- Roads / Track ----------------
        roads = self._parse_roads(config)
        road_map = RoadMap(roads)
        self.road_track = RoadTrack(road_map)

        # ---------------- TF ----------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------------- Dynamic registry ----------------
        # active robots with goals: { name: {'goal': (x,y,yaw), 'path_pub': ..., 'rviz_pub': ...} }
        self.active: Dict[str, Dict] = {}

        # Fleet manager instance (rebuilt when robot set changes)
        self.fm_tree: Optional[FleetManagerTree] = None
        self._rebuild_fm_tree()  # initial empty

        # ---------------- Danger areas publisher ----------------
        self.danger_pub = self.create_publisher(MarkerArray, "danger_areas", 10)

        # ---------------- Services ----------------
        self.plan_srv = self.create_service(PlanPath, "plan_path", self._on_plan_path)
        self.cancel_srv = self.create_service(CancelPath, "cancel_path", self._on_cancel_path)

        # ---------------- Timer ----------------
        self.timer = self.create_timer(self.timer_period, self._on_timer)
        self.get_logger().info("Planner ready (services: /plan_path, /cancel_path)")

    # ------------------------------------------------------------------ #
    # Service callbacks
    # ------------------------------------------------------------------ #

    def _on_plan_path(self, req: PlanPath.Request, res: PlanPath.Response):
        name = req.robot_name.strip()
        if not name:
            res.accepted = False
            res.message = "robot_name is empty"
            return res

        goal: PoseYaw = (float(req.x), float(req.y), float(req.yaw))

        # Create pubs for this robot if first time
        if name not in self.active:
            path_pub = self.create_publisher(Path2D, f"{name}/path", 1)
            rviz_pub = self.create_publisher(Path, f"{name}/path_for_rviz", 1)
            self.active[name] = {"goal": goal, "path_pub": path_pub, "rviz_pub": rviz_pub}
            self._rebuild_fm_tree()  # robot set changed → rebuild FM tree
            self.get_logger().info(f"➕ Registered robot '{name}'")
        else:
            # Update goal only
            self.active[name]["goal"] = goal
            self.get_logger().info(f"Updated goal for '{name}' → ({goal[0]:.2f}, {goal[1]:.2f}, yaw={goal[2]:.2f})")

        res.accepted = True
        res.message = "Goal accepted"
        return res

    def _on_cancel_path(self, req: CancelPath.Request, res: CancelPath.Response):
        name = req.robot_name.strip()
        if not name or name not in self.active:
            res.ok = False
            res.message = f"Robot '{name}' is not active"
            return res

        # Publish empty path to stop the tracker, then remove robot
        empty = Path2D()
        empty.header.frame_id = self.frame_id
        self.active[name]["path_pub"].publish(empty)

        # Remove from registry and rebuild FM
        del self.active[name]
        self._rebuild_fm_tree()
        self.get_logger().info(f"Cancelled and removed robot '{name}'")

        res.ok = True
        res.message = "Cancelled"
        return res

    # ------------------------------------------------------------------ #
    # Timer (continuous replanning)
    # ------------------------------------------------------------------ #

    def _on_timer(self) -> None:
        if not self.active:
            return

        # 1) Build pose/goal dict from TF + active registry
        robot_poses_goals = {}
        for name, entry in self.active.items():
            pose = self._lookup_pose(name)
            if pose is None:
                self.get_logger().warn(f"[{name}] TF unavailable; skipping this cycle for this robot")
                continue
            robot_poses_goals[name] = {"pose": pose, "goal": entry["goal"]}

        if not robot_poses_goals or self.fm_tree is None:
            return

        # 2) Plan/tick
        self.fm_tree.set_robot_poses_goals(robot_poses_goals)
        self.fm_tree.tick()
        robots_to_move = set(self.fm_tree.get_robots_to_move())

        # 3) Publish paths
        now = self.get_clock().now()
        for name, entry in self.active.items():
            path_for_robot = None
            if name in robots_to_move:
                path_for_robot = self.fm_tree.get_pose_path(name)  # [(x,y,dir), ...]

            p2d, pnav = self._build_path_msgs(path_for_robot, now, name)
            entry["path_pub"].publish(p2d)
            entry["rviz_pub"].publish(pnav)

        # 4) Publish danger areas (if your FM tree provides them)
        da = self.fm_tree.get_danger_areas()
        self._publish_danger_areas(da, now)

    # ------------------------------------------------------------------ #
    # FM tree rebuild on robot set changes
    # ------------------------------------------------------------------ #

    def _rebuild_fm_tree(self) -> None:
        names = list(self.active.keys())
        vehicles = [EdyMobile() for _ in names]  # simple: same type; extend if you want per-robot types
        self.fm_tree = FleetManagerTree(len(names), self.road_track, vehicles, names)
        self.fm_tree.setup()
        self.get_logger().info(f"Rebuilt FleetManagerTree with robots={names}")

    # ------------------------------------------------------------------ #
    # Helpers: YAML, TF, messages, danger areas
    # ------------------------------------------------------------------ #

    def _resolve_yaml_path(self, cfg_param: str) -> str:
        if cfg_param:
            if os.path.isabs(cfg_param):
                return cfg_param
            pkg = get_package_share_directory("em_vehicle_control")
            return os.path.join(pkg, "config", cfg_param)
        # default
        pkg = get_package_share_directory("em_vehicle_control")
        return os.path.join(pkg, "config", "roads.yaml")

    @staticmethod
    def _load_yaml(path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"YAML not found: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def _get_param_or_yaml(self, key: str, yaml_root: dict, default=None):
        param = self.get_parameter(key)
        if param.type_ != rclpy.parameter.Parameter.Type.NOT_SET:
            # returns native Python type (str/float/int/etc.)
            return param.value
        return yaml_root.get(key, default)

    def _parse_roads(self, yaml_root: dict) -> List[RoadSegment]:
        roads_def = yaml_root.get("roads", [])
        if not roads_def:
            raise RuntimeError("YAML must contain a 'roads' list.")
        roads: List[RoadSegment] = []
        for i, r in enumerate(roads_def):
            try:
                x1, y1, x2, y2 = float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])
                roads.append(RoadSegment((x1, y1), (x2, y2)))
            except Exception as e:
                raise RuntimeError(f"Invalid roads[{i}] entry: {r} ({e})")
        return roads

    def _lookup_pose(self, robot_name: str) -> Optional[PoseYaw]:
        try:
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                self.frame_id, f"{robot_name}/base_link", rclpy.time.Time(), Duration(seconds=0.5)
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.TransformException):
            return None
        t = tf.transform
        yaw = R.from_quat([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]).as_euler("zyx", degrees=False)[0]
        return (t.translation.x, t.translation.y, yaw)

    def _build_path_msgs(self, path, now, robot_name: str):
        path2d = Path2D()
        path2d.header = Header(stamp=self._to_time_msg(now), frame_id=self.frame_id)

        nav_path = Path()
        nav_path.header = Header(stamp=self._to_time_msg(now), frame_id=self.frame_id)

        if not path:
            return path2d, nav_path

        # final yaw sourced from stored goal
        goal_yaw = self.active.get(robot_name, {}).get("goal", (0.0, 0.0, 0.0))[2]

        for x_, y_, dirflag in path:
            x, y, d = float(x_), float(y_), int(dirflag)

            p2 = Pose2D()
            p2.x, p2.y, p2.theta = x, y, 0.0
            p2.direction_flag = Pose2D.FORWARD if d == 1 else Pose2D.BACKWARD
            p2.header.stamp = self._to_time_msg(now)
            p2.header.frame_id = self.frame_id
            path2d.poses.append(p2)

            ps = PoseStamped()
            ps.header.stamp = self._to_time_msg(now)
            ps.header.frame_id = self.frame_id
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            nav_path.poses.append(ps)

        path2d.poses[-1].theta = float(goal_yaw)
        return path2d, nav_path

    @staticmethod
    def _to_time_msg(t) -> TimeMsg:
        sec, nsec = t.seconds_nanoseconds()
        return TimeMsg(sec=sec, nanosec=nsec)

    def _publish_danger_areas(self, polygons: List[Polygon], now) -> None:
        mlist = MarkerArray()
        mid = 0

        # Filled
        for poly in polygons:
            tris = triangulate(poly)
            if not tris:
                continue
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = self._to_time_msg(now)
            m.ns = "danger_areas_filled"
            m.id = mid; mid += 1
            m.type = Marker.TRIANGLE_LIST
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color = ColorRGBA(r=1.0, g=0.3, b=0.3, a=0.25)
            for tri in tris:
                x0, y0 = tri.exterior.coords[0]
                x1, y1 = tri.exterior.coords[1]
                x2, y2 = tri.exterior.coords[2]
                for (x, y) in [(x0, y0), (x1, y1), (x2, y2)]:
                    p = RosPoint(); p.x = float(x); p.y = float(y); p.z = 0.0
                    m.points.append(p)
            mlist.markers.append(m)

        # Outlines
        for poly in polygons:
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = self._to_time_msg(now)
            m.ns = "danger_areas_outline"
            m.id = mid; mid += 1
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.02
            m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9)
            for x, y in poly.exterior.coords:
                p = RosPoint(); p.x = float(x); p.y = float(y); p.z = 0.0
                m.points.append(p)
            if m.points:
                m.points.append(m.points[0])
            mlist.markers.append(m)

        self.danger_pub.publish(mlist)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Planner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
