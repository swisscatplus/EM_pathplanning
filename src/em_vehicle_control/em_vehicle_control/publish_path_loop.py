import rclpy
from rclpy.node import Node
from em_vehicle_control_msgs.msg import Path2D, Pose2D

class LoopingPathPublisher(Node):
    def __init__(self):
        super().__init__('looping_path_publisher')
        self.publisher = self.create_publisher(Path2D, 'path', 10)
        self.timer_period = 10.0  # seconds between path switches
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.current_path_id = 5
        self.direction = 1  # 1 = FORWARD, -1 = BACKWARD

        self.poses_coordinates = {
            5: [(0.0, 0.0), (0.0, 0.5)],
            6: [(0.0, 0.5), (0.0, 0.0)]
        }

        self.get_logger().info("LoopingPathPublisher initialized. Starting loop between path 5 and 6.")

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
        self.get_logger().info(f"Published path {self.current_path_id} ({'FORWARD' if self.direction == 1 else 'BACKWARD'})")

        # Alternate path for next cycle
        if self.current_path_id == 5:
            self.current_path_id = 6
            self.direction = -1
        else:
            self.current_path_id = 5
            self.direction = 1

def main(args=None):
    rclpy.init(args=args)
    node = LoopingPathPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
