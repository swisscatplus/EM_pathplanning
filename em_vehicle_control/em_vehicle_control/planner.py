from scipy.spatial.transform import Rotation as R
import networkx as nx
import numpy as np
import time

from em_vehicle_control.helper_classes.map import RoadSegment, RoadMap, RoadTrack
from em_vehicle_control.helper_classes.RNBTree import FleetManagerTree
from em_vehicle_control.helper_classes.vehicles import EdyMobile, Edison

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, PoseStamped
from em_vehicle_control_msgs.msg import Pose2D, Path2D
from std_msgs.msg import Header
from nav_msgs.msg import Path
import tf2_ros
from rclpy.duration import Duration
from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from shapely.geometry import Polygon
from shapely.ops import triangulate

from py_trees.common import BlackBoxLevel
from py_trees.display import render_dot_tree


class Planner(Node):
    """
    Node that runs periodically
    Takes current poses, passes them through the planner,
    outputs path message with poses and the timestamp when the vehicle is allowed to pass the node
    """

    def __init__(self):
        super().__init__("planner")
        timer_period = 2  # seconds

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        test_roads = [
            RoadSegment((2.87, 1.67), (3.52, 13.67)),
            RoadSegment((6.9, 0.3), (7.55, 15.67)),
            RoadSegment((4.72, 1.67), (5.7, 8.64)),
            RoadSegment((2.87, 8), (7.55, 8.64)),
            RoadSegment((2.87, 1.67), (7.55, 2.32)),
            # RoadSegment((0,0), (10,10))
        ]
        test_map = RoadMap(test_roads)
        test_graph = RoadTrack(test_map)
        # self.robot_names = ["robot_0", "robot_1", "robot_2", "robot_3"]
        # self.goals = [(3.354,3.865,1.57), (6.03,8.46,0),(3,4.32, 0), (3.5,2.1,1.57)]#[(3.354,3.865,1.57), (3,4.32, 0)]#[(3,4.32, 0)] #
        # self.vehicles = [EdyMobile(), EdyMobile(), EdyMobile(), EdyMobile()]
        self.robot_names = ["robot_0","robot_1"]
        self.goals = [(3.3,3.17,0),(3.3,4.28,0)]#[(3.354,3.865,1.57), (3,4.32, 0)]#[(3,4.32, 0)] #
        self.vehicles = [EdyMobile(),EdyMobile()]
        self.FMTree = FleetManagerTree(len(self.robot_names), test_graph, self.vehicles, self.robot_names)
        self.FMTree.setup()
        self.timer = self.create_timer(timer_period, self.plan_callback)
        self.pubbers = []
        self.rviz_pubbers = []
        for robot_name in self.robot_names:
            pub = self.create_publisher(Path2D, f"{robot_name}/path", 1)
            self.pubbers.append(pub)
            pub = self.create_publisher(Path, f"{robot_name}/path_for_rviz", 1)
            self.rviz_pubbers.append(pub)
        self.DA_pubber = self.create_publisher(MarkerArray, '/danger_areas', 10)

    def plan_callback(self):
        robot_poses_goals = {}
        for robot_name, robot_goal in zip(self.robot_names, self.goals):
            try:
                # Lookup transform from source_frame to target_frame
                transform: TransformStamped = self.tf_buffer.lookup_transform(
                    "world",  # Target frame
                    f"{robot_name}/base_link",  # Source frame
                    rclpy.time.Time(),
                    Duration(seconds=1.0),
                )
                x = transform.transform.translation.x
                y = transform.transform.translation.y
                qx = transform.transform.rotation.x
                qy = transform.transform.rotation.y
                qz = transform.transform.rotation.z
                qw = transform.transform.rotation.w
                r = R.from_quat([qx, qy, qz, qw])
                yaw = r.as_euler("zyx", degrees=False)[0]
                robot_poses_goals[robot_name] = {
                    "pose": (x,y,yaw),
                    "goal": robot_goal
                }
                
            except tf2_ros.LookupException as e:
                self.get_logger().error(f"Transform lookup failed: {e}")
                return

            except tf2_ros.ExtrapolationException as e:
                self.get_logger().error(f"Extrapolation exception: {e}")
                return

            except tf2_ros.TransformException as e:
                self.get_logger().error(f"Transform exception: {e}")
                return
        if robot_poses_goals == {}:
            return
        self.FMTree.set_robot_poses_goals(robot_poses_goals)
        self.FMTree.tick()
        robots_to_move = self.FMTree.get_robots_to_move()
        paths = []
        for robot in self.robot_names:
            if not robot in robots_to_move:
                paths.append(None)
                continue
            path = self.FMTree.get_pose_path(robot)
            paths.append(path)

        current_time = rclpy.time.Time()
        # self.get_logger().info(f'Paths: {paths}, {robots_to_move}')
       
        for path, pub, rviz_pub in zip(paths, self.pubbers, self.rviz_pubbers):
            path_2D_msg = Path2D()
            path_2D_msg.header.frame_id = "world"
            path_msg = Path()
            path_msg.header.frame_id = "world"
            if path is None:
                pub.publish(path_2D_msg)
                rviz_pub.publish(path_msg)
                continue
            for node in path:
                pose_2D_msg = Pose2D()
                pose_2D_msg.x = float(node[0])
                pose_2D_msg.y = float(node[1])
                if node[2] == 1:
                    pose_2D_msg.direction_flag = Pose2D.FORWARD
                else:
                    pose_2D_msg.direction_flag = Pose2D.BACKWARD
                pose_2D_msg.header.stamp = Time(sec=current_time.seconds_nanoseconds()[0], nanosec=current_time.seconds_nanoseconds()[1])
                pose_2D_msg.header.frame_id = "world"
                path_2D_msg.poses.append(pose_2D_msg)
                pose_msg = PoseStamped()
                pose_msg.pose.position.x = float(node[0])
                pose_msg.pose.position.y = float(node[1])
                pose_msg.pose.position.z = 0.0
                pose_msg.pose.orientation.w = 1.0
                pose_msg.pose.orientation.x = 0.0
                pose_msg.pose.orientation.y = 0.0
                pose_msg.pose.orientation.z = 0.0
                path_msg.poses.append(pose_msg)
            pub.publish(path_2D_msg)
            rviz_pub.publish(path_msg)
        DA = self.FMTree.get_danger_areas()
        self.publish_danger_areas(DA)

    def publish_danger_areas(self, danger_areas):
        """Publish all danger areas as a MarkerArray."""
        marker_array = MarkerArray()
        for idx, polygon in enumerate(danger_areas):
            marker = self.shapely_polygon_to_marker(polygon, idx)
            marker_array.markers.append(marker)

        self.DA_pubber.publish(marker_array)

    def shapely_polygon_to_marker(self, polygon, marker_id):
        """Convert a Shapely Polygon to an RViz Marker."""
        marker = Marker()
        marker.header.frame_id = "world"  # Set the frame ID (e.g., "world" or "map")
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "danger_areas"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP  # For filling the polygon
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.01

        # Marker color (outline)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0  # Fully opaque line

        # Add points for the exterior boundary
        for coord in polygon.exterior.coords:
            point = Point()
            point.x = coord[0]
            point.y = coord[1]
            point.z = 0.0
            marker.points.append(point)

        # Close the loop
        if len(polygon.exterior.coords) > 0:
            marker.points.append(marker.points[0])

        return marker


def main(args=None):
    rclpy.init(args=args)
    node = Planner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
