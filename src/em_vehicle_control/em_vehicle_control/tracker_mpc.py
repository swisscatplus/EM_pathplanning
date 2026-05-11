# ros2 run em_vehicle_control publish_path


from scipy.spatial.transform import Rotation as R
import math
import threading
import numpy as np
from typing import Tuple

from em_vehicle_control.path_registry import HardcodedPathProvider
from em_vehicle_control.helper_classes.mpc_tracker_theta import MPCTracker
from em_vehicle_control.helper_classes.segment import *

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, Twist, Quaternion
from em_vehicle_control_msgs.action import ExecutePath
from em_vehicle_control_msgs.msg import Pose2D, Path2D
from rclpy.duration import Duration
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

PosePt2D = Tuple[float, float, float]  # (x, y, yaw) values

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
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_link_frame", "base_link")
        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.base_link_frame = (
            self.get_parameter("base_link_frame").get_parameter_value().string_value
        )
        self.declare_parameter("tf_timeout_sec", 0.02)
        self.tf_timeout_sec = (
            self.get_parameter("tf_timeout_sec").get_parameter_value().double_value
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub_cmd_vel = self.create_publisher(Twist, f"cmd_vel", 10)

        ##############
        # Parameters #
        ##############
        self.path = None
        self.path = None
        self.path_msg_lock = threading.RLock()
        self.path_provider = HardcodedPathProvider()

        self._goal_pending = False
        self._active_goal_handle = None
        self._active_path_id = None
        self._active_action_done = threading.Event()
        self._active_action_status = None
        self._active_action_message = ""
        self._last_distance_to_goal = math.inf
        self._last_action_state = ExecutePath.Feedback.STATE_ACCEPTED
        self._last_action_state_label = "accepted"

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

        self.action_callback_group = ReentrantCallbackGroup()
        self.execute_path_server = ActionServer(
            self,
            ExecutePath,
            "execute_path",
            execute_callback=self.execute_path_callback,
            goal_callback=self.execute_path_goal_callback,
            cancel_callback=self.execute_path_cancel_callback,
            callback_group=self.action_callback_group,
        )

        self.get_logger().info("✅ Tracker node initialized.")
        self.get_logger().info(
            "Action server ready on /execute_path with path IDs: "
            + ", ".join(str(path_id) for path_id in self.path_provider.available_path_ids())
        )

    def path_subscription(self, msg):
        with self.path_msg_lock:
            if self._active_goal_handle is not None:
                self.get_logger().warn(
                    "Ignoring topic path while an execute_path action is active."
                )
                return
            self.path = msg.poses
            self.tracker.initialise_new_path()

    def execute_path_goal_callback(self, goal_request):
        path_id = int(goal_request.path_id)
        with self.path_msg_lock:
            if not self.path_provider.has_path(path_id):
                self.get_logger().warn(f"Rejecting unknown path_id={path_id}.")
                return GoalResponse.REJECT

            if self._goal_pending or self._active_goal_handle is not None or self.path:
                self.get_logger().warn(
                    f"Rejecting path_id={path_id}; tracker is already executing a path."
                )
                return GoalResponse.REJECT

            self._goal_pending = True

        self.get_logger().info(f"Accepted execute_path goal for path_id={path_id}.")
        return GoalResponse.ACCEPT

    def execute_path_cancel_callback(self, goal_handle):
        with self.path_msg_lock:
            if not self._same_goal_handle(goal_handle, self._active_goal_handle):
                return CancelResponse.REJECT
            self._cancel_active_action_locked("Action goal canceled by client.")
        return CancelResponse.ACCEPT

    def execute_path_callback(self, goal_handle):
        path_id = int(goal_handle.request.path_id)
        result = ExecutePath.Result()
        result.path_id = path_id

        with self.path_msg_lock:
            self._goal_pending = False
            if self._active_goal_handle is not None or self.path:
                result.success = False
                result.message = "Tracker is already executing a path."
                goal_handle.abort()
                return result

            if not self.path_provider.has_path(path_id):
                result.success = False
                result.message = f"Unknown path_id={path_id}."
                goal_handle.abort()
                return result

            path_msg = self.path_provider.build_path(
                path_id,
                stamp=self.get_clock().now().to_msg(),
                frame_id=self.map_frame,
            )
            self.path = path_msg.poses
            self.tracker.initialise_new_path()

            self._active_goal_handle = goal_handle
            self._active_path_id = path_id
            self._active_action_status = None
            self._active_action_message = ""
            self._active_action_done.clear()
            self._set_action_feedback_locked(
                math.inf,
                ExecutePath.Feedback.STATE_ACCEPTED,
                "accepted",
            )

        self.get_logger().info(f"Executing path {path_id}.")

        while rclpy.ok():
            if self._active_action_done.wait(timeout=0.25):
                break
            if goal_handle.is_cancel_requested:
                with self.path_msg_lock:
                    self._cancel_active_action_locked("Action goal canceled by client.")
                break
            self._publish_action_feedback(goal_handle)

        with self.path_msg_lock:
            status = self._active_action_status
            message = self._active_action_message

            if status == "succeeded":
                goal_handle.succeed()
                result.success = True
                result.message = message or f"Path {path_id} reached goal."
            elif status == "canceled":
                goal_handle.canceled()
                result.success = False
                result.message = message or "Action goal canceled."
            else:
                goal_handle.abort()
                result.success = False
                result.message = message or "Action goal aborted."

            self._clear_active_action_locked(goal_handle)

        return result

    def _set_action_feedback_locked(self, distance_to_goal, state, state_label):
        self._last_distance_to_goal = float(distance_to_goal)
        self._last_action_state = int(state)
        self._last_action_state_label = state_label

    def _publish_action_feedback(self, goal_handle):
        with self.path_msg_lock:
            if not self._same_goal_handle(goal_handle, self._active_goal_handle):
                return

            feedback = ExecutePath.Feedback()
            feedback.path_id = int(self._active_path_id)
            feedback.distance_to_goal = (
                self._last_distance_to_goal
                if math.isfinite(self._last_distance_to_goal)
                else -1.0
            )
            feedback.state = int(self._last_action_state)
            feedback.state_label = self._last_action_state_label

        goal_handle.publish_feedback(feedback)

    def _complete_active_action_locked(self, status, message):
        if self._active_goal_handle is None:
            return
        self._active_action_status = status
        self._active_action_message = message
        self._active_action_done.set()

    def _cancel_active_action_locked(self, message):
        if self._active_goal_handle is None:
            return
        self.pub_twist(0.0, 0.0)
        self.path = None
        self._set_action_feedback_locked(
            0.0,
            ExecutePath.Feedback.STATE_CANCELING,
            "canceling",
        )
        self._complete_active_action_locked("canceled", message)

    def _clear_active_action_locked(self, goal_handle):
        if not self._same_goal_handle(goal_handle, self._active_goal_handle):
            return
        self._active_goal_handle = None
        self._active_path_id = None
        self._active_action_status = None
        self._active_action_message = ""
        self._active_action_done.clear()
        self._set_action_feedback_locked(
            math.inf,
            ExecutePath.Feedback.STATE_ACCEPTED,
            "accepted",
        )

    @staticmethod
    def _same_goal_handle(left, right):
        if left is right:
            return True
        if left is None or right is None:
            return False
        try:
            return left.goal_id.uuid == right.goal_id.uuid
        except AttributeError:
            return False

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
                self.map_frame,
                self.base_link_frame,
                rclpy.time.Time(),
                Duration(seconds=self.tf_timeout_sec),
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
        if len(self.path) < 2:
            self.get_logger().warn("Received a path with fewer than two poses; clearing it.")
            self.pub_twist(0.0, 0.0)
            self.path = None
            self._complete_active_action_locked("aborted", "Path has fewer than two poses.")
            return

        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return

        segments = create_segments(self.path)
        distance_to_goal = math.hypot(
            robot_pose[0] - self.path[-1].x,
            robot_pose[1] - self.path[-1].y,
        )
        self._set_action_feedback_locked(
            distance_to_goal,
            ExecutePath.Feedback.STATE_TRACKING,
            "tracking",
        )

        # Check if robot has reached goal (both pos & angle) or is just at final pos
        goal_reached, angle_diff = self.tracker.check_goal(robot_pose, segments, return_angle=True)

        if goal_reached:
            self.get_logger().info("✅ Goal reached! Stopping and clearing path.")
            self.pub_twist(0.0, 0.0)
            self.path = None
            self._set_action_feedback_locked(
                0.0,
                ExecutePath.Feedback.STATE_REACHED,
                "reached",
            )
            if self._active_goal_handle is not None:
                self._complete_active_action_locked(
                    "succeeded",
                    f"Path {self._active_path_id} reached goal.",
                )
            return

        # Only rotate if robot is at the goal position (but not aligned)
        if self.tracker.is_at_goal_position(robot_pose, segments) and angle_diff > self.tracker.goal_angle_tol:
            self._set_action_feedback_locked(
                distance_to_goal,
                ExecutePath.Feedback.STATE_FINAL_ALIGN,
                "final_align",
            )
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
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
