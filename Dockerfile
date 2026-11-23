FROM ros:jazzy-ros-base
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ros-gz ros-jazzy-cv-bridge python3-opencv python3-colcon-common-extensions \
    libgl1-mesa-dri libegl-mesa0 mesa-utils \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /ws
COPY src /ws/src
RUN source /opt/ros/jazzy/setup.bash && colcon build --symlink-install
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "line_follower", "line_follow.launch.py"]
