#!/bin/bash
set -e

export ROS_DOMAIN_ID=10

# Ensure Fast DDS profile path is active (optional but safe)
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/root/.ros/fastdds.xml

# Source ROS2 environment
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

# Execute your launch file with specified robot_name
#exec ros2 launch em_vehicle_control tracker_loop.launch.py
exec ros2 launch em_vehicle_control multi_robot.launch.py \
  map_config:=/path/to/roads.yaml \
  robots_config:=/path/to/robots.yaml