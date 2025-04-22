import rclpy
from rclpy.node import Node
from em_vehicle_control_msgs.msg import Path2D, Pose2D


class SimplePathPublisher(Node):
    def __init__(self):
        super().__init__('simple_path_publisher')
        self.publisher = self.create_publisher(Path2D, 'path', 10)

        # Publish the same path 10 times (once per second)
        self.counter = 0
        self.max_repeats = 10
        self.timer = self.create_timer(1.0, self.timer_callback)

        # Define the path once
        self.msg = Path2D()
        pose_start = Pose2D()
        pose_start.x = 0.0
        pose_start.y = 0.0

        pose_end = Pose2D()
        pose_end.x = 0.0
        pose_end.y = 1.0

        self.msg.poses.append(pose_start)
        self.msg.poses.append(pose_end)

    def timer_callback(self):
        if self.counter < self.max_repeats:
            self.publisher.publish(self.msg)
            self.get_logger().info(f"Published path ({self.counter+1}/{self.max_repeats})")
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


if __name__ == '__main__':
    main()
