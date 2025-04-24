import rclpy
from rclpy.node import Node
from em_vehicle_control_msgs.msg import Path2D, Pose2D


class SimplePathPublisher(Node):
    def __init__(self):
        super().__init__('simple_path_publisher')
        self.publisher = self.create_publisher(Path2D, 'path', 10)

        # Publish the same path 10 times (once per second)
        self.counter = 0
        self.max_repeats = 1
        self.timer = self.create_timer(1.0, self.timer_callback)

        self.msg = Path2D()

        poses_coordinates = [
            (0.0, 0.0),
            (0.5, 0.0),
            (1.0, 0.5),
            (1.0, 1.0)
        ]

        for x, y in poses_coordinates:
            pose = Pose2D()
            pose.x = x
            pose.y = y
            pose.direction_flag = Pose2D.FORWARD
            self.msg.poses.append(pose)

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
