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

        poses_coordinates_1 = [
            (0.4875, 2.985),
            (0.4875, 1.985),
            (0.4875, 0.985),
            (0.4875, 0.0),
            (0.4875, -0.325),
            (0.65, -0.4875),
            (1.65, -0.4875),
            (2.65, -0.4875),
            (3.65, -0.4875),
            (4.06, -0.4875),
            (4.35, -0.325),
            (4.35, 0.0),
            (4.35, 0.885)
        ]

        poses_coordinates_2 = [
            (4.35, 0.885),
            (4.35, 0.0),
            (4.35, -0.325),
            (4.06, -0.4875),
            (3.65, -0.4875),
            (2.65, -0.4875),
            (1.65, -0.4875),
            (0.65, -0.4875),
            (0.4875, -0.325),
            (0.4875, 0.0),
            (0.4875, 0.985),
            (0.4875, 1.985),
            (0.4875, 2.985)
        ]


        poses_coordinates = [
            (0.0, 0.0),
            (-0.5, 0.0),
        ]

        # poses_coordinates = [
        #     (-0.5, 0.0),
        #     (0.0, 0.0),
        # ]

        # poses_coordinates = [
        #     (0.49, 3.225),
        #     (0.49, 2.225),
        #     (0.49, 1.225),
        #     (0.49, 0.0),
        #     (0.65, -0.30),
        #     (1.00, -0.45),
        #     (2.00, -0.45),
        #     (3.00, -0.45),
        #     (4.03, -0.45),
        #     (4.10, -0.30),
        #     (4.19, -0.15),
        #     (4.19, 0.00),
        #     (4.19, 1.00),
        #     (4.19, 2.00),
        # ]

        # poses_coordinates = [
        #     (4.526, 2.338),
        #     (4.526, 2.838),
        #     (4.526, 3.338),
        #     (4.526, 3.838),
        #     (4.526, 4.338),
        #     (4.526, 4.518),
        #     (4.526, 5.518),
        #     (4.426, 5.750),
        #     (4.300, 5.750),
        #     (4.174, 5.518),
        #     (4.174, 4.518),
        #     (4.174, 3.518),
        #     (4.174, 2.518),
        #     (4.174, 2.338),
        # ]

        for x, y in poses_coordinates_2:
            pose = Pose2D()
            pose.x = x
            pose.y = y
            #pose.direction_flag = Pose2D.BACKWARD
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
