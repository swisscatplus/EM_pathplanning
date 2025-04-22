from scipy.spatial.transform import Rotation as R
import networkx as nx
import numpy as np
import time

from em_vehicle_control.helper_classes.map import RoadSegment, RoadMap, RoadGraph, RoadTrack
from em_vehicle_control.helper_classes.path_planner import PathPlanner, PathPointDatum
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
        self.path_planner = PathPlanner(test_map, test_graph, 1)
        self.robot_names = ["robot_0"]
        self.goals = [(3,4.32, 0)]#[(7.23, 14.9)]
        self.vehicles = [EdyMobile()]
        # self.timer = self.create_timer(timer_period, self.plan_callback)
        self.pubbers = []
        self.rviz_pubbers = []
        for robot_name in self.robot_names:
            pub = self.create_publisher(Path2D, f"path", 1)
            self.pubbers.append(pub)
            pub = self.create_publisher(Path, f"path_for_rviz", 1)
            self.rviz_pubbers.append(pub)

    def plan_callback(self):
        current_time = rclpy.time.Time()

        paths = [[
            # PathPointDatum(4.429, 7.54, 0, 1),
            # PathPointDatum(4.429, 4.729, 0, 1),
            PathPointDatum(4.227, 4.729, 0, 1),
            PathPointDatum(4.429, 4.729, 0, -1),
            PathPointDatum(4.429, 7.54, 0, -1),
        ]]

        # tight loop at NMR
        self.get_logger().info("Sending path")
        # paths = [[
        #     PathPointDatum(4.292, 7.445, 0 ,1),
        #     PathPointDatum(4.298, 7.215, 0 ,1),
        #     PathPointDatum(4.519, 7.216, 0 ,1),
        #     PathPointDatum(4.495, 7.5, 0 ,1),
        #     PathPointDatum(4.292, 7.445, 0 ,1)
        # ]]
       
        for path, pub, rviz_pub in zip(paths, self.pubbers, self.rviz_pubbers):
            if path is None:
                continue
            path_2D_msg = Path2D()
            path_2D_msg.header.frame_id = "map"
            path_msg = Path()
            path_msg.header.frame_id = "map"
            for node in path:
                pose_2D_msg = Pose2D()
                pose_2D_msg.x = float(node.x)
                pose_2D_msg.y = float(node.y)
                if node.direction == 1:
                    pose_2D_msg.direction_flag = Pose2D.FORWARD
                else:
                    pose_2D_msg.direction_flag = Pose2D.BACKWARD
                time_in_nanoseconds = rclpy.time.Duration(nanoseconds=int(node.time * 1e9))
                pose_2D_msg.header.stamp = (time_in_nanoseconds + current_time).to_msg()
                pose_2D_msg.header.frame_id = "world"
                path_2D_msg.poses.append(pose_2D_msg)
                pose_msg = PoseStamped()
                pose_msg.pose.position.x = float(node.x)
                pose_msg.pose.position.y = float(node.y)
                pose_msg.pose.position.z = 0.0
                pose_msg.pose.orientation.w = 1.0
                pose_msg.pose.orientation.x = 0.0
                pose_msg.pose.orientation.y = 0.0
                pose_msg.pose.orientation.z = 0.0
                path_msg.poses.append(pose_msg)
            pub.publish(path_2D_msg)
            rviz_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = Planner()

    try:
        # rclpy.spin(node)
        time.sleep(2)
        node.plan_callback()
        rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
