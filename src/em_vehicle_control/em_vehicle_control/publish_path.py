# ros2 service call /switch_path em_vehicle_control_msgs/srv/SwitchPath "{path_id: 1, direction: -1}"

import rclpy
from rclpy.node import Node

from em_vehicle_control.path_registry import YamlPathProvider
from em_vehicle_control_msgs.msg import Path2D, Pose2D
from em_vehicle_control_msgs.srv import SwitchPath


class SimplePathPublisher(Node):
    def __init__(self):
        super().__init__("simple_path_publisher")
        self.declare_parameter("paths_config", "paths.yaml")
        self.declare_parameter("max_repeats", 1)

        self.path_provider = YamlPathProvider(self.get_parameter("paths_config").value)
        self.publisher = self.create_publisher(Path2D, "path", 10)

        self.counter = 0
        self.max_repeats = int(self.get_parameter("max_repeats").value)
        self.msg = Path2D()
        self.current_path_id = None

        self.timer = self.create_timer(1.0, self.timer_callback)
        self.timer.cancel()

        self.srv = self.create_service(SwitchPath, "switch_path", self.switch_path_callback)
        self.get_logger().info(
            "SimplePathPublisher initialized with path IDs: "
            + ", ".join(str(path_id) for path_id in self.path_provider.available_path_ids())
        )

    @staticmethod
    def _direction_override(direction):
        if int(direction) == 1:
            return Pose2D.FORWARD
        if int(direction) == -1:
            return Pose2D.BACKWARD
        return None

    def switch_path_callback(self, request, response):
        path_id = int(request.path_id)
        direction = self._direction_override(request.direction)

        if not self.path_provider.has_path(path_id) or direction is None:
            msg = f"Invalid path ID or direction: path_id={path_id}, direction={request.direction}"
            self.get_logger().warn(msg)
            response.success = False
            response.message = msg
            return response

        self.msg = self.path_provider.build_path(
            path_id,
            stamp=self.get_clock().now().to_msg(),
            direction_override=direction,
        )
        self.current_path_id = path_id
        self.counter = 0
        if self.timer.is_canceled():
            self.timer = self.create_timer(1.0, self.timer_callback)
        else:
            self.timer.reset()

        msg = f"Switched to path {path_id} with direction {'FORWARD' if direction == Pose2D.FORWARD else 'BACKWARD'}"
        self.get_logger().info(msg)
        response.success = True
        response.message = msg
        return response

    def timer_callback(self):
        if self.counter < self.max_repeats:
            self.publisher.publish(self.msg)
            self.get_logger().info(
                f"Published path {self.current_path_id} ({self.counter + 1}/{self.max_repeats})"
            )
            self.counter += 1
        else:
            self.get_logger().info("Done publishing path.")
            self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = SimplePathPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
