import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from em_vehicle_control.path_registry import PathWorkflow, YamlPathProvider, msg_value_to_direction
from em_vehicle_control_msgs.msg import Path2D


class LoopingPathPublisher(Node):
    def __init__(self):
        super().__init__("looping_path_publisher")

        self.declare_parameter("paths_config", "paths.yaml")
        self.declare_parameter("workflow_config", "path_workflows.yaml")
        self.declare_parameter("workflow", "")
        self.declare_parameter("start_immediately", None)

        self.path_provider = YamlPathProvider(self.get_parameter("paths_config").value)
        self.workflow = PathWorkflow(
            self.get_parameter("workflow_config").value,
            self.get_parameter("workflow").value,
        )
        start_param = self.get_parameter("start_immediately").value
        self.start_immediately = self.workflow.start_immediately if start_param is None else bool(start_param)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.publisher = self.create_publisher(Path2D, "path", qos)

        self.step_index = 0
        self.get_logger().info(
            f"Loaded paths from {self.path_provider.config_path}: "
            + ", ".join(str(path_id) for path_id in self.path_provider.available_path_ids())
        )
        self.get_logger().info(
            f"Using workflow '{self.workflow.name}' from {self.workflow.config_path} "
            f"with {len(self.workflow.steps)} steps."
        )

        self._wait_for_subscriber()

        if self.start_immediately:
            self.timer_callback()

        self.timer = self.create_timer(self.workflow.timer_period_s, self.timer_callback)

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

    def publish_current_and_advance(self):
        step = self.workflow.steps[self.step_index]
        path_id = step["path_id"]
        msg = self.path_provider.build_path(
            path_id,
            stamp=self.get_clock().now().to_msg(),
            direction_override=step["direction"],
            densify_ds=self.workflow.densify_ds,
        )
        self.publisher.publish(msg)
        self.get_logger().info(
            f"Published workflow '{self.workflow.name}' step {self.step_index + 1}/"
            f"{len(self.workflow.steps)}: path {path_id} "
            f"({msg_value_to_direction(msg.poses[0].direction_flag)})"
        )
        self.step_index = (self.step_index + 1) % len(self.workflow.steps)

    def timer_callback(self):
        self.publish_current_and_advance()


def main(args=None):
    rclpy.init(args=args)
    node = LoopingPathPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
