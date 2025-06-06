# ros2 run em_vehicle_control publish_path


from scipy.spatial.transform import Rotation as R
import networkx as nx
from shapely import LineString, Point, MultiLineString, MultiPoint
import threading
import numpy as np
from typing import Tuple, Union, List
from enum import Enum
from time import time

from em_vehicle_control.helper_classes.mpc_tracker_theta import MPCTracker
from em_vehicle_control.helper_classes.segment import *

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, Transform, Twist, Quaternion
from em_vehicle_control_msgs.msg import Pose2D, Path2D
import tf2_ros
from rclpy.duration import Duration
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA

PosePt2D = Tuple[float, float, float]  # (x, y, yaw) values

class Direction(Enum):
    FORWARD = 1
    BACKWARD = -1

class Tracker(Node):
    """
    This node will manage a single robot.
    It will receive the pose of its own robot as quickly as it can,
    and set command vel appropriately based on the path it has stored.
    The path is updated whenever there is a new path message
    """

    def __init__(self):
        super().__init__("tracker")
        self.declare_parameter("robot_name", "unnamed_robot")
        self.robot_name = (
            self.get_parameter("robot_name").get_parameter_value().string_value
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub_cmd_vel = self.create_publisher(Twist, f"cmd_vel", 10)

        ##############
        # Parameters #
        ##############
        self.path = None
        self.path = None
        self.path_msg_lock = threading.Lock()

        self.path_subscription = self.create_subscription(
            Path2D, f"path", self.path_subscription, 1
        )
        self.path_subscription
        self.timer = self.create_timer(0.125, self.timer_callback)  # 8Hz

        #########
        # WRITE #
        #########
        self.write = False
        if self.write:
            with open("src/data.txt", "w") as file:
                file.write(
                    f"xR, yR, yaw_R, xA, yA, xB, yB, curr_cte, curr_oe, v, omega \n"
                )

        #########################
        # Plot MPC plan in RVIZ #
        #########################
        self.plot_rviz = True
        self.tracker = MPCTracker(plot_rviz = self.plot_rviz)
        if self.plot_rviz:
            self.marker_publisher = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)

        self.get_logger().info("✅ Tracker node initialized.")

    def path_subscription(self, msg):
        with self.path_msg_lock:
            self.path = msg.poses
            self.tracker.initialise_new_path()

    def pub_twist(self, v: float, omega: float) -> None:
        """
        v: linear velocity
        omega: angular velocity
        """
        twist_msg = Twist()
        twist_msg.linear.x = v
        twist_msg.angular.z = omega
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        self.pub_cmd_vel.publish(twist_msg)

    def get_robot_pose(self) -> PosePt2D:
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                "map",  # Target frame, I changed it to map, it's world in Jasper's version
                #f"{self.robot_name}/base_link",  # Source frame
                f"base_link",
                rclpy.time.Time(),
                Duration(seconds=1.0),
            )
        except:
            return None
        if transform is None or transform.transform is None:
            return None
        robot_pose = transform.transform
        qx = robot_pose.rotation.x
        qy = robot_pose.rotation.y
        qz = robot_pose.rotation.z
        qw = robot_pose.rotation.w
        r = R.from_quat([qx, qy, qz, qw])
        robot_yaw = r.as_euler("zyx", degrees=False)[0]
        robot_pose = (
            robot_pose.translation.x,
            robot_pose.translation.y,
            robot_yaw
        )
        return robot_pose

    def control_loop(self):
        """
        Control loop that manages and runs the MPC tracker,
        and publishes velocity commands. When the robot reaches the final position
        but is not aligned with the final heading, it rotates in place.
        """
        if self.path is None or self.path == []:
            self.pub_twist(0.0, 0.0)
            return

        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return

        segments = create_segments(self.path)

        # Check if robot has reached goal (both pos & angle) or is just at final pos
        goal_reached, angle_diff = self.tracker.check_goal(robot_pose, segments, return_angle=True)

        if goal_reached:
            self.get_logger().info("✅ Goal reached! Stopping and clearing path.")
            self.pub_twist(0.0, 0.0)
            self.path = None
            return

        # Only rotate if robot is at the goal position (but not aligned)
        if self.tracker.is_at_goal_position(robot_pose, segments) and angle_diff > self.tracker.goal_angle_tol:
            if self.tracker.final_theta is None:
                self.tracker.compute_final_theta(segments)
            final_theta = self.tracker.final_theta
            angle_error = (final_theta - robot_pose[2] + np.pi) % (2 * np.pi) - np.pi
            gain = 2.0  # Tune this gain as needed
            omega = np.clip(gain * angle_error, -self.tracker.omega_max, self.tracker.omega_max)
            self.pub_twist(0.0, omega)
            return

        # Otherwise, use MPC to track path
        if not self.plot_rviz:
            v, omega = self.tracker.track(robot_pose, segments)
        else:
            marker_array = MarkerArray()
            v, omega, x, y, theta = self.tracker.track(robot_pose, segments)
            for i, (x_, y_, theta_) in enumerate(zip(x, y, theta)):
                marker = self.create_arrow_marker(x_, y_, robot_pose[2] + theta_, i)
                marker_array.markers.append(marker)
            self.marker_publisher.publish(marker_array)

        self.pub_twist(v, omega)

    def timer_callback(self) -> None:
        """
        Callback that runs every tick of the ros timer. 
        Locks path object for thread safety.
        """
        with self.path_msg_lock:
            self.control_loop()
    
    def yaw_to_quaternion(self, yaw):
        """Convert a yaw angle (in radians) to a quaternion."""
        r = R.from_euler('z', yaw, degrees=False)
        quat_array = r.as_quat()  # This returns [qx, qy, qz, qw]
        quat_msg = Quaternion()
        quat_msg.x = quat_array[0]
        quat_msg.y = quat_array[1]
        quat_msg.z = quat_array[2]
        quat_msg.w = quat_array[3]
        return quat_msg

    def create_arrow_marker(self, x, y, yaw, marker_id):
        """Create a marker for RViz to visualize an arrow at (x, y) with the given yaw."""
        marker = Marker()
        marker.header.frame_id = f'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'arrows'
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Set the pose of the arrow (position and orientation)
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.pose.orientation = self.yaw_to_quaternion(yaw)

        # Set the scale of the arrow (size)
        marker.scale.x = 0.2  # Arrow length
        marker.scale.y = 0.01  # Arrow width
        marker.scale.z = 0.01  # Arrow height

        # Set the color of the arrow (RGBA)
        marker.color = ColorRGBA(r=0.25, g=0.88, b=0.82, a=0.8)  # turqoise arrow

        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()  # Infinite duration
        return marker
        
    def delete_all_markers(self):
        """Create a marker to delete all previous markers."""
        marker = Marker()
        marker.header.frame_id = f'/base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.action = Marker.DELETEALL
        return marker



def main(args=None):
    rclpy.init(args=args)
    node = Tracker()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
