FROM ros:humble-ros-base-jammy
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    gnupg \
    libgl1 \
    lsb-release \
    python3 \
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    python3-yaml \
    ros-${ROS_DISTRO}-geometry-msgs \
    ros-${ROS_DISTRO}-nav-msgs \
    ros-${ROS_DISTRO}-rclpy \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    ros-${ROS_DISTRO}-std-msgs \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    cvxpy \
    matplotlib \
    networkx \
    "numpy<2.0" \
    opencv-python \
    py_trees \
    Rtree \
    scipy \
    shapely \
    transitions

RUN mkdir -p /root/.ros
COPY config/fastdds.xml /root/.ros/fastdds.xml
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/root/.ros/fastdds.xml

WORKDIR /ros2_ws
COPY src src
RUN rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    . /opt/ros/${ROS_DISTRO}/setup.sh && \
    colcon build --symlink-install --packages-select em_vehicle_control_msgs em_vehicle_control

COPY docker/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "em_vehicle_control", "tracker_loop.launch.py"]
