FROM nvblox-docker:amd64

SHELL ["/bin/bash", "-o", "pipefail", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy
ENV SENSOR_WS=/opt/sensors_ws
ENV RGBD_WS=/rgbd_camera_intel_dev
ENV ML_VENV=/opt/ml_venv

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV ROS_DOMAIN_ID=0
ENV ROS_LOCALHOST_ONLY=0
ENV CYCLONEDDS_URI=file:///etc/cyclonedds/cyclonedds.xml

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
        python3-venv \
        python3-argcomplete \
        python3-colcon-common-extensions \
        python3-rosdep \
        python3-vcstool \
        python3-dotenv \
        python3-requests \
        libusb-1.0-0 \
        libusb-1.0-0-dev \
        udev \
        usbutils \
        v4l-utils \
        libssl-dev \
        libglfw3-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libasio-dev \
        libtinyxml2-dev \
        libopencv-dev \
        python3-opencv \
        libpcap-dev \
        libpcl-dev \
        pcl-tools \
        at \
        ros-${ROS_DISTRO}-xacro \
        ros-${ROS_DISTRO}-tf2-ros \
        ros-${ROS_DISTRO}-tf2-tools \
        ros-${ROS_DISTRO}-tf-transformations \
        ros-${ROS_DISTRO}-robot-state-publisher \
        ros-${ROS_DISTRO}-teleop-twist-keyboard \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
        ros-${ROS_DISTRO}-cyclonedds \
        ros-${ROS_DISTRO}-diagnostic-updater \
        ros-${ROS_DISTRO}-diagnostic-msgs \
        ros-${ROS_DISTRO}-sensor-msgs \
        ros-${ROS_DISTRO}-sensor-msgs-py \
        ros-${ROS_DISTRO}-std-srvs \
        ros-${ROS_DISTRO}-nav-msgs \
        ros-${ROS_DISTRO}-geometry-msgs \
        ros-${ROS_DISTRO}-image-transport \
        ros-${ROS_DISTRO}-image-transport-plugins \
        ros-${ROS_DISTRO}-compressed-image-transport \
        ros-${ROS_DISTRO}-compressed-depth-image-transport \
        ros-${ROS_DISTRO}-camera-info-manager \
        ros-${ROS_DISTRO}-cv-bridge \
        ros-${ROS_DISTRO}-image-geometry \
        ros-${ROS_DISTRO}-navigation2 \
        ros-${ROS_DISTRO}-nav2-bringup \
        ros-${ROS_DISTRO}-nav2-costmap-2d \
        ros-${ROS_DISTRO}-nav2-rviz-plugins \
        ros-${ROS_DISTRO}-robot-localization \
        ros-${ROS_DISTRO}-topic-tools \
        ros-${ROS_DISTRO}-slam-toolbox \
        ros-${ROS_DISTRO}-pointcloud-to-laserscan \
        ros-${ROS_DISTRO}-pcl-conversions \
        ros-${ROS_DISTRO}-pcl-ros \
        ros-${ROS_DISTRO}-rviz2 \
        ros-${ROS_DISTRO}-rqt \
        ros-${ROS_DISTRO}-rqt-common-plugins \
        ros-${ROS_DISTRO}-rqt-image-view; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    rosdep init || true; \
    rosdep fix-permissions || true; \
    rosdep update || true

# Optional ML env for ClipSeg / Open3D
ENV PATH=${ML_VENV}/bin:$PATH
ENV PYTHONPATH=${ML_VENV}/lib/python3.12/site-packages:/usr/lib/python3/dist-packages:/opt/ros/${ROS_DISTRO}/lib/python3.12/site-packages
RUN set -eux; \
    python3 -m venv --system-site-packages ${ML_VENV}; \
    ${ML_VENV}/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel; \
    ${ML_VENV}/bin/python -m pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128; \
    ${ML_VENV}/bin/python -m pip install --no-cache-dir \
        --index-url https://pypi.org/simple \
        "numpy<2" \
        pillow \
        packaging \
        pyyaml \
        requests \
        regex \
        filelock \
        safetensors \
        "tokenizers>=0.19,<0.20" \
        psutil \
        "huggingface_hub>=0.23.0" \
        "empy==3.3.4" \
        catkin_pkg \
        lark \
        open3d \
        "transformers==4.41.2" \
        "accelerate>=0.31.0"

# Sensor workspace
ARG REALSENSE_ROS_REF=4.56.1
ARG VELODYNE_REF=ros2

WORKDIR ${SENSOR_WS}
RUN mkdir -p src

RUN set -eux; \
    cd ${SENSOR_WS}/src; \
    git clone https://github.com/IntelRealSense/realsense-ros.git; \
    cd realsense-ros; \
    git checkout ${REALSENSE_ROS_REF}

RUN set -eux; \
    cd ${SENSOR_WS}/src; \
    git clone https://github.com/ros-drivers/velodyne.git; \
    cd velodyne; \
    git checkout ${VELODYNE_REF}

# RUN set -eux; \
#     cd ${SENSOR_WS}/src; \
#     git clone -b ros2 https://github.com/KumarRobotics/ublox.git; \
#     git clone https://github.com/Robeff-Technology/ntrip_client.git

RUN set -ex; \
    source /opt/ros/${ROS_DISTRO}/setup.bash; \
    if [ -f /workspaces/isaac_ros-dev/install/setup.bash ]; then source /workspaces/isaac_ros-dev/install/setup.bash; fi; \
    rosdep update || true; \
    rosdep install -i --from-paths src --rosdistro ${ROS_DISTRO} -y \
        --skip-keys=" \
            librealsense2 \
            librealsense2-dev \
            launch_pytest \
            launch_testing \
            launch_testing_ament_cmake \
            ament_lint_common \
            python3-tqdm \
            xacro \
            libopencv-dev \
            libopencv-contrib-dev \
            libopencv-imgproc-dev \
            python-opencv \
            python3-opencv \
            libpcap0.8-dev \
            python3-requests \
        "; \
    colcon build --symlink-install \
        --packages-select \
            realsense2_camera_msgs \
            realsense2_camera \
            velodyne_msgs \
            velodyne_driver \
            velodyne_pointcloud \
            velodyne_laserscan \
            velodyne \
            # ublox \
            # ublox_msgs \
            # ublox_gps \
            # ublox_serialization \
            # ntrip_client \
        --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_TESTING=OFF

RUN mkdir -p /etc/cyclonedds ${RGBD_WS}/src

RUN set -eux; \
    echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc; \
    echo "if [ -f /workspaces/isaac_ros-dev/install/setup.bash ]; then source /workspaces/isaac_ros-dev/install/setup.bash; fi" >> /root/.bashrc; \
    echo "source ${SENSOR_WS}/install/setup.bash" >> /root/.bashrc; \
    echo "if [ -f ${RGBD_WS}/install/setup.bash ]; then source ${RGBD_WS}/install/setup.bash; fi" >> /root/.bashrc; \
    echo "export PATH=${ML_VENV}/bin:\$PATH" >> /root/.bashrc; \
    echo "export PYTHONPATH=${ML_VENV}/lib/python3.12/site-packages:/usr/lib/python3/dist-packages:/opt/ros/${ROS_DISTRO}/lib/python3.12/site-packages:\$PYTHONPATH" >> /root/.bashrc; \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /root/.bashrc; \
    echo "export ROS_DOMAIN_ID=0" >> /root/.bashrc; \
    echo "export ROS_LOCALHOST_ONLY=0" >> /root/.bashrc

WORKDIR /workspaces/isaac_ros-dev
CMD ["bash"]