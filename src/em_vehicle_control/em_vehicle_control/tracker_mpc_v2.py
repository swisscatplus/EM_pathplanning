from __future__ import annotations

import math
import threading
from enum import Enum
from typing import Optional, Tuple, List

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, Twist, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

from em_vehicle_control.helper_classes.mpc_tracker_theta import MPCTracker
from em_vehicle_control.helper_classes.segment import create_segments
from em_vehicle_control_msgs.msg import Pose2D, Path2D


PosePt2D = Tuple[float, float, float]  # (x, y, yaw)


class Mode(Enum):
    TRACK = 1       # MPC tracking the path
    FINAL_SPIN = 2  # At goal position; only align heading gently
    DONE = 3        # Goal reached; stop & clear path


class Tracker(Node):
    """
    ROS 2 node controlling a single robot with an MPC path tracker.

    Stabilizations against "almost-at-goal oscillation":
      - Hysteresis between entering/leaving the goal position radius
      - Debounce: require N consecutive ticks with pos+angle ok to finish
      - Gentler final-spin controller (P on heading with capped omega)
      - Mild low-pass smoothing on v/omega
    """

    def __init__(self) -> None:
        super().__init__("tracker")

        # --- add in __init__ ---
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_link_frame", "")
        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        # Default base_link_frame: if empty, derive from robot_name
        bl_param = self.get_parameter("base_link_frame").get_parameter_value().string_value
        self.base_link_frame = bl_param if bl_param else f"{self.robot_name}/base_link"

        # ---------- Parameters ----------
        self.declare_parameter("robot_name", "unnamed_robot")
        self.robot_name = self.get_parameter("robot_name").get_parameter_value().string_value

        # Frequency (increase if your MPC and stack can keep up)
        self.control_dt = 0.05  # 20 Hz
        self.timer = self.create_timer(self.control_dt, self._on_timer)

        # ---------- TF / Pub ----------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pub_cmd_vel = self.create_publisher(Twist, "cmd_vel", 10)

        # ---------- Path / Tracker ----------
        self._path: Optional[List[Pose2D]] = None
        self._path_lock = threading.Lock()

        # RViz trajectory preview (arrows)
        self.plot_rviz: bool = True
        self.tracker = MPCTracker(plot_rviz=self.plot_rviz)
        if self.plot_rviz:
            self.marker_pub = self.create_publisher(MarkerArray, "visualization_marker_array", 10)

        # Subscribe to path updates
        self.path_sub = self.create_subscription(Path2D, "path", self._on_path, 1)

        # ---------- Goal handling / stabilization ----------
        # Tolerances (tune to your platform)
        self.mode: Mode = Mode.TRACK
        self.goal_pos_tol_enter: float = 0.12   # meters; enter FINAL_SPIN when inside
        self.goal_pos_tol_exit: float = 0.16    # meters; leave FINAL_SPIN if outside (hysteresis)
        # Try to read tracker's angle tol; fallback to 6 degrees
        self.goal_angle_tol: float = getattr(self.tracker, "goal_angle_tol", 6.0 * math.pi / 180.0)

        # Final spin behavior (gentler than generic omega_max)
        self.final_spin_gain: float = 1.2
        self.final_spin_omega_max: float = 0.6  # rad/s

        # Debounce: require these many consecutive ticks with pos & angle OK
        self.goal_debounce_needed: int = 6
        self._goal_debounce_count: int = 0

        # Command smoothing (0=no smoothing, closer to 1=more smoothing)
        self._v_prev: float = 0.0
        self._w_prev: float = 0.0
        self.cmd_alpha: float = 0.5

        self.get_logger().info("✅ Tracker node initialized.")

    # --------------------------------------------------------------------- #
    # Subscriptions / Callbacks
    # --------------------------------------------------------------------- #

    def _on_path(self, msg: Path2D) -> None:
        """Receive a new path and (re)initialize tracking state."""
        with self._path_lock:
            self._path = msg.poses
            self.tracker.initialise_new_path()
            self.mode = Mode.TRACK
            self._goal_debounce_count = 0
            self._v_prev = 0.0
            self._w_prev = 0.0

    def _on_timer(self) -> None:
        """Timer callback; protected path access."""
        with self._path_lock:
            self._control_loop()

    # --------------------------------------------------------------------- #
    # Core control
    # --------------------------------------------------------------------- #

    def _control_loop(self) -> None:
        # No path → hold still
        if not self._path:
            self._publish_twist(0.0, 0.0)
            return

        # Current robot pose
        robot_pose = self._get_robot_pose()
        if robot_pose is None:
            return

        # Build segments for MPC
        segments = create_segments(self._path)

        # Goal XYθ (use last pose in the path)
        goal = self._goal_xytheta()
        if goal is None:
            self._publish_twist(0.0, 0.0)
            return
        gx, gy, gtheta = goal

        # Distance to goal position
        dx = robot_pose[0] - gx
        dy = robot_pose[1] - gy
        dist = math.hypot(dx, dy)

        # Desired final heading from tracker (if available)
        if getattr(self.tracker, "final_theta", None) is None:
            self.tracker.compute_final_theta(segments)
        final_theta = getattr(self.tracker, "final_theta", gtheta)

        # Heading error
        angle_err = self._wrap_angle(final_theta - robot_pose[2])

        # Debounce bookkeeping
        pos_ok = dist <= self.goal_pos_tol_enter
        ang_ok = abs(angle_err) <= self.goal_angle_tol
        if pos_ok and ang_ok:
            self._goal_debounce_count += 1
        else:
            # Looser reset in FINAL_SPIN, strict in TRACK
            if (self.mode == Mode.FINAL_SPIN and dist > self.goal_pos_tol_exit) or self.mode == Mode.TRACK:
                self._goal_debounce_count = 0

        # ---------- State transitions ----------
        if self.mode == Mode.TRACK:
            if dist <= self.goal_pos_tol_enter:
                self.mode = Mode.FINAL_SPIN

        elif self.mode == Mode.FINAL_SPIN:
            if dist > self.goal_pos_tol_exit:
                self.mode = Mode.TRACK
            elif self._goal_debounce_count >= self.goal_debounce_needed:
                self.mode = Mode.DONE

        elif self.mode == Mode.DONE:
            self.get_logger().info("✅ Goal reached! Stopping and clearing path.")
            self._publish_twist(0.0, 0.0)
            self._path = None
            return

        # ---------- Actions ----------
        if self.mode == Mode.TRACK:
            if not self.tracker.plot_rviz:
                v_cmd, w_cmd = self.tracker.track(robot_pose, segments)
            else:
                v_cmd, w_cmd, xs, ys, thetas = self.tracker.track(robot_pose, segments)
                self._publish_preview_markers(xs, ys, thetas)

        elif self.mode == Mode.FINAL_SPIN:
            # Freeze translation, gently align heading
            w_cmd = np.clip(self.final_spin_gain * angle_err,
                            -self.final_spin_omega_max, self.final_spin_omega_max)
            v_cmd = 0.0

        # Smooth and publish
        v_cmd, w_cmd = self._smooth_cmd(v_cmd, w_cmd)
        self._publish_twist(v_cmd, w_cmd)

    # --------------------------------------------------------------------- #
    # Utilities
    # --------------------------------------------------------------------- #

    def _get_robot_pose(self) -> Optional[PosePt2D]:
        """Lookup robot pose in 'map' frame and return (x, y, yaw)."""
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.map_frame,  # was "map"
                self.base_link_frame,  # was "base_link"
                rclpy.time.Time(),
                Duration(seconds=0.5),
            )
        except Exception:
            return None

        if transform is None or transform.transform is None:
            return None

        t = transform.transform
        qx, qy, qz, qw = t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w
        yaw = R.from_quat([qx, qy, qz, qw]).as_euler("zyx", degrees=False)[0]
        return (t.translation.x, t.translation.y, yaw)

    def _goal_xytheta(self) -> Optional[Tuple[float, float, float]]:
        """Return final (x, y, theta) from the current path."""
        if not self._path:
            return None
        p = self._path[-1]
        # Path2D.Pose2D may have fields x, y, theta
        theta = getattr(p, "theta", 0.0)
        return (p.x, p.y, theta)

    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Wrap angle to (-pi, pi]."""
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def _smooth_cmd(self, v: float, w: float) -> Tuple[float, float]:
        """First-order low-pass on commands to reduce jitter."""
        a = self.cmd_alpha
        v_out = a * self._v_prev + (1.0 - a) * v
        w_out = a * self._w_prev + (1.0 - a) * w
        self._v_prev, self._w_prev = v_out, w_out
        return v_out, w_out

    def _publish_twist(self, v: float, w: float) -> None:
        msg = Twist()
        msg.linear.x = float(v)
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(w)
        self.pub_cmd_vel.publish(msg)

    # ---------------- RViz helpers ---------------- #

    def _yaw_to_quat(self, yaw: float) -> Quaternion:
        """Convert yaw (rad) to quaternion."""
        r = R.from_euler("z", yaw, degrees=False).as_quat()  # [x, y, z, w]
        q = Quaternion()
        q.x, q.y, q.z, q.w = float(r[0]), float(r[1]), float(r[2]), float(r[3])
        return q

    def _publish_preview_markers(self, xs: List[float], ys: List[float], thetas: List[float]) -> None:
        if not self.plot_rviz or self.marker_pub is None:
            return
        marray = MarkerArray()
        now = self.get_clock().now().to_msg()
        for i, (x, y, th) in enumerate(zip(xs, ys, thetas)):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = now
            m.ns = "mpc_preview"
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD

            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.0
            m.pose.orientation = self._yaw_to_quat(float(th))

            m.scale.x = 0.2   # arrow length
            m.scale.y = 0.02  # arrow width
            m.scale.z = 0.02  # arrow height
            m.color = ColorRGBA(r=0.25, g=0.88, b=0.82, a=0.9)  # turquoise
            m.lifetime = Duration(seconds=0.0).to_msg()  # persistent
            marray.markers.append(m)

        self.marker_pub.publish(marray)

    # Optional: delete all markers (not used, but handy)
    def _delete_all_markers(self) -> None:
        if not self.plot_rviz or self.marker_pub is None:
            return
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.action = Marker.DELETEALL
        self.marker_pub.publish(MarkerArray(markers=[m]))


# ------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------- #

def main(args=None) -> None:
    rclpy.init(args=args)
    node = Tracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
