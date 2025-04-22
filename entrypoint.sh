#!/bin/bash
set -e

# Source ROS2 environment
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

# Execute your launch file with specified robot_name
exec ros2 launch em_vehicle_control tracker.launch.py robot_name:=robot1
