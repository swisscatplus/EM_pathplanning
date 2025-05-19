# Use the official ROS 2 Humble base image
FROM ros:humble-ros-base

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV ROS_DOMAIN_ID=10

# Install ROS & system packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    git \
    curl \
    lsb-release \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install \
    scipy \
    numpy \
    shapely \
    networkx \
    cvxpy

# === ADD FASTDDS CONFIG FILE ===
RUN mkdir -p /root/.ros
COPY fastdds.xml /root/.ros/fastdds.xml
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/root/.ros/fastdds.xml

# Create and set workspace directory
RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

# Copy local ROS2 packages into container
COPY src /ros2_ws/src

# Install ROS2 package dependencies (no need for `rosdep init` in official base image)
RUN rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"

# Automatically source ROS 2 and workspace on shell startup
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]
