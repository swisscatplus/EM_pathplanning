import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from em_vehicle_control_msgs.msg import Path2D, Pose2D

class LoopingPathPublisher(Node):
    def __init__(self):
        super().__init__('looping_path_publisher')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,  # works with Fix 1
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.publisher = self.create_publisher(Path2D, 'path', qos)

        self.timer_period = 60.0
        self.current_path_id = 1
        self.direction = 1  # 1 = FORWARD, -1 = BACKWARD

        self.poses_coordinates = {
            1: [
                (0.325, 2.985), (0.325, 1.985), (0.325, 0.985), (0.325, 0.0),
                (0.325, -0.200), (0.65, -0.325), (1.65, -0.325), (2.65, -0.325),
                (3.65, -0.325), (4.06, -0.325), (4.35, -0.200), (4.35, 0.0), (4.35, 0.885)
            ],
            2: [
                (4.35, 0.885), (4.35, 0.0), (4.35, -0.200), (4.06, -0.325),
                (3.65, -0.325), (2.65, -0.325), (1.65, -0.325), (0.65, -0.325),
                (0.325, -0.200), (0.325, 0.0), (0.325, 0.985), (0.325, 1.985), (0.325, 2.985)
            ],
        }

        self.get_logger().info("LoopingPathPublisher initialized. Starting loop between path 1 and 2.")

        # Wait for at least one subscriber before first publish
        self._wait_for_subscriber()

        # Publish immediately once a subscriber is present
        self.timer_callback()

        # Then switch to periodic publishing every 80s
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def _wait_for_subscriber(self, timeout_sec: float = 10.0):
        # Spin until a subscriber is present or timeout elapses (optional timeout)
        import time
        start = time.time()
        while self.publisher.get_subscription_count() == 0:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start > timeout_sec:
                self.get_logger().warn("No subscribers yet; continuing anyway.")
                break
        if self.publisher.get_subscription_count() > 0:
            self.get_logger().info("Subscriber detected on 'path'; publishing first message now.")

    def load_path(self, path_id, direction):
        msg = Path2D()
        for x, y in self.poses_coordinates[path_id]:
            pose = Pose2D()
            pose.x = x
            pose.y = y
            pose.direction_flag = Pose2D.FORWARD if direction == 1 else Pose2D.BACKWARD
            msg.poses.append(pose)
        return msg

    def timer_callback(self):
        msg = self.load_path(self.current_path_id, self.direction)
        self.publisher.publish(msg)
        self.get_logger().info(
            f"Published path {self.current_path_id} "
            f"({'FORWARD' if self.direction == 1 else 'BACKWARD'})"
        )

        # Alternate path for next cycle
        if self.current_path_id == 1:
            self.current_path_id = 2
            self.direction = -1
        else:
            self.current_path_id = 1
            self.direction = 1

def main(args=None):
    rclpy.init(args=args)
    node = LoopingPathPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
