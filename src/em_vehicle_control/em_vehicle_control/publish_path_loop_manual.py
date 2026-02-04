import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_srvs.srv import Trigger  # <-- pour le service
from em_vehicle_control_msgs.msg import Path2D, Pose2D

import math

# ==== CONFIG ICI ====
MANUAL_MODE = True  # False = auto avec timer, True = manuel via service
# ros2 service call /next_path std_srvs/srv/Trigger {}
DENSIFY_DS = 0.02   # spacing (m) between points after densification
# ====================


def densify_path(points, ds=0.03):
    """
    Insert intermediate points along each segment so spacing is about ds meters.
    Keeps original vertices and adds linearly-interpolated points between them.
    """
    if not points:
        return []

    dense = [points[0]]
    for p0, p1 in zip(points[:-1], points[1:]):
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            continue

        n = int(dist / ds)
        for i in range(1, n + 1):
            t = i / (n + 1)
            dense.append((p0[0] + t * dx, p0[1] + t * dy))

        dense.append(p1)

    return dense


class LoopingPathPublisher(Node):
    def __init__(self):
        super().__init__('looping_path_publisher')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.publisher = self.create_publisher(Path2D, 'path', qos)

        self.manual_mode = MANUAL_MODE
        self.timer_period = 80.0
        self.current_path_id = 1
        self.direction = 1  # 1 = FORWARD, -1 = BACKWARD

        # -----------------------------
        # RAW PATHS (your geometry)
        # -----------------------------
        self.poses_coordinates_raw = {
            1: [
                # LEFT VERTICAL CRUISE (safe/centered)
                (0.425, 2.985), (0.425, 1.985), (0.425, 0.985),
                (0.425, 0.700), (0.425, 0.500),

                # LAST 0.5 m SHIFT to TRUE START LINE x=0.525
                (0.445, 0.400), (0.475, 0.300), (0.505, 0.200),
                (0.520, 0.100), (0.525, 0.0),

                # SMOOTH TURN to bottom centerline
                (0.540, -0.08), (0.570, -0.16), (0.610, -0.24),
                (0.650, -0.325),

                # BOTTOM STRAIGHT
                (1.65, -0.325), (2.65, -0.325), (3.65, -0.325), (4.00, -0.325),

                # SMOOTH TURN up to RIGHT CRUISE VERTICAL x=4.3
                (4.08, -0.31), (4.16, -0.26), (4.22, -0.20),
                (4.27, -0.12), (4.30, 0.0),

                # RIGHT VERTICAL CRUISE (stay at 4.3 almost all the way)
                (4.30, 0.200), (4.30, 0.385),

                # LAST 0.5 m SHIFT to TRUE GOAL LINE x=4.2
                (4.28, 0.485), (4.25, 0.585), (4.22, 0.685),
                (4.205, 0.785), (4.20, 0.885),
            ],

            2: [
                # reverse with same logic
                (4.20, 0.885),
                (4.205, 0.785), (4.22, 0.685), (4.25, 0.585), (4.28, 0.485),
                (4.30, 0.385), (4.30, 0.200), (4.30, 0.0),

                (4.27, -0.12), (4.22, -0.20), (4.16, -0.26), (4.08, -0.31),
                (4.00, -0.325), (3.65, -0.325), (2.65, -0.325), (1.65, -0.325),
                (0.650, -0.325),

                (0.610, -0.24), (0.570, -0.16), (0.540, -0.08),
                (0.525, 0.0),

                (0.520, 0.100), (0.505, 0.200), (0.475, 0.300), (0.445, 0.400),
                (0.425, 0.500), (0.425, 0.700), (0.425, 0.985),
                (0.425, 1.985), (0.425, 2.985),
            ],
        }

        # -----------------------------
        # DENSIFIED PATHS (for MPC)
        # -----------------------------
        self.poses_coordinates = {
            pid: densify_path(pts, ds=DENSIFY_DS)
            for pid, pts in self.poses_coordinates_raw.items()
        }

        self.get_logger().info(
            f"Densified paths with ds={DENSIFY_DS:.3f} m. "
            f"Sizes: " +
            ", ".join([f"{pid}:{len(self.poses_coordinates[pid])}" for pid in self.poses_coordinates])
        )

        mode_str = "MANUAL (service /next_path)" if self.manual_mode else "AUTO (timer)"
        self.get_logger().info(f"LoopingPathPublisher initialized in {mode_str} mode.")

        # Attendre un subscriber
        self._wait_for_subscriber()

        if self.manual_mode:
            # Mode manuel : on crée un service, pas de timer
            self.srv = self.create_service(Trigger, 'next_path', self.next_path_callback)
            self.get_logger().info(
                "Manual mode: call 'ros2 service call /next_path std_srvs/srv/Trigger {}' "
                "pour publier le trajet suivant."
            )
        else:
            # Mode auto : on publie une fois, puis toutes les 80s
            self.timer_callback()
            self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def _wait_for_subscriber(self, timeout_sec: float = 10.0):
        import time
        start = time.time()
        while self.publisher.get_subscription_count() == 0:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start > timeout_sec:
                self.get_logger().warn("No subscribers yet; continuing anyway.")
                break
        if self.publisher.get_subscription_count() > 0:
            self.get_logger().info("Subscriber detected on 'path'.")

    def load_path(self, path_id, direction):
        msg = Path2D()
        for x, y in self.poses_coordinates[path_id]:
            pose = Pose2D()
            pose.x = x
            pose.y = y
            pose.direction_flag = Pose2D.FORWARD if direction == 1 else Pose2D.BACKWARD
            msg.poses.append(pose)
        return msg

    def publish_current_and_advance(self):
        """Publie le trajet courant puis prépare le suivant."""
        msg = self.load_path(self.current_path_id, self.direction)
        self.publisher.publish(msg)
        self.get_logger().info(
            f"Published path {self.current_path_id} "
            f"({'FORWARD' if self.direction == 1 else 'BACKWARD'})"
        )

        # Alterner pour le prochain appel
        if self.current_path_id == 1:
            self.current_path_id = 2
            self.direction = -1
        else:
            self.current_path_id = 1
            self.direction = 1

    def timer_callback(self):
        # Utilise la même logique que le service
        self.publish_current_and_advance()

    def next_path_callback(self, request, response):
        """Callback du service Trigger : publie le prochain trajet à la demande."""
        self.publish_current_and_advance()
        response.success = True
        response.message = "Next path triggered."
        return response


def main(args=None):
    rclpy.init(args=args)
    node = LoopingPathPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
