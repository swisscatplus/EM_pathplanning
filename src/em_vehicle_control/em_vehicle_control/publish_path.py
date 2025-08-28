# ros2 service call /switch_path em_vehicle_control_msgs/srv/SwitchPath "{path_id: 1, direction: -1}"

import rclpy
from rclpy.node import Node
from em_vehicle_control_msgs.msg import Path2D, Pose2D
from em_vehicle_control_msgs.srv import SwitchPath

class SimplePathPublisher(Node):
    def __init__(self):
        super().__init__('simple_path_publisher')
        self.publisher = self.create_publisher(Path2D, 'path', 10)

        self.counter = 0
        self.max_repeats = 1
        self.msg = Path2D()

        self.current_path_id = None

        # Predefined paths
        self.poses_coordinates = {
            1: [
                (0.430, 2.985), (0.430, 1.985), (0.430, 0.985), (0.430, 0.0),
                (0.430, -0.200), (0.65, -0.430), (1.65, -0.430), (2.65, -0.430),
                (3.65, -0.430), (4.06, -0.430), (4.35, -0.200), (4.35, 0.0), (4.35, 0.885)
            ],
            2: [
                (4.35, 0.885), (4.35, 0.0), (4.35, -0.200), (4.06, -0.430),
                (3.65, -0.430), (2.65, -0.430), (1.65, -0.430), (0.65, -0.430),
                (0.430, -0.200), (0.430, 0.0), (0.430, 0.985), (0.430, 1.985), (0.430, 2.985)
            ],
            3: [
                (4.252, 1.695), (4.252, 2.695), (4.252, 3.695), (4.252, 4.695), (4.252, 5.695)
            ],
            4: [
                (4.252, 5.695), (4.252, 4.695), (4.252, 3.695), (4.252, 2.695), (4.252, 1.695)
            ],
            5: [
                (0.0, 0.0), (-0.5, 0.0)
            ],
            6: [
                (-0.5, 0.0), (0.0, 0.0)
            ]
        }

        # Setup timer but cancel it until path is selected
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.timer.cancel()

        # Create switch_path service
        self.srv = self.create_service(SwitchPath, 'switch_path', self.switch_path_callback)

        self.get_logger().info("SimplePathPublisher initialized. Waiting for service call to select path.")

    def load_path(self, coordinates, direction):
        self.msg.poses.clear()
        for x, y in coordinates:
            pose = Pose2D()
            pose.x = x
            pose.y = y
            pose.direction_flag = Pose2D.FORWARD if direction == 1 else Pose2D.BACKWARD
            self.msg.poses.append(pose)

    def switch_path_callback(self, request, response):
        path_id = request.path_id
        direction = request.direction  # 1 for FORWARD, -1 for BACKWARD

        if path_id in self.poses_coordinates and direction in [1, -1]:
            self.load_path(self.poses_coordinates[path_id], direction)
            self.current_path_id = path_id
            self.counter = 0
            if self.timer.is_canceled():
                self.timer = self.create_timer(1.0, self.timer_callback)
            else:
                self.timer.reset()
            msg = f"Switched to path {path_id} with direction {'FORWARD' if direction == 1 else 'BACKWARD'}"
            self.get_logger().info(msg)
            response.success = True
            response.message = msg
        else:
            msg = f"Invalid path ID or direction: path_id={path_id}, direction={direction}"
            self.get_logger().warn(msg)
            response.success = False
            response.message = msg
        return response

    def timer_callback(self):
        if self.counter < self.max_repeats:
            self.publisher.publish(self.msg)
            if self.current_path_id is not None:
                self.get_logger().info(f"Published path {self.current_path_id} ({self.counter + 1}/{self.max_repeats})")
            else:
                self.get_logger().info(f"Published path ({self.counter + 1}/{self.max_repeats})")
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
