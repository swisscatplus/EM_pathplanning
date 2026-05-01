#!/bin/bash
set -e

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/root/.ros/fastdds.xml}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-10}"

source /opt/ros/"$ROS_DISTRO"/setup.bash

if [ -f /ros2_ws/install/setup.bash ]; then
  source /ros2_ws/install/setup.bash
fi

exec "$@"
