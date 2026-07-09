# GraphNav-VLM
Semantic navigation system for a UGV using ROS 2, RGB-D/LiDAR perception, open-vocabulary segmentation, VLM scene descriptions, LLM-based candidate selection, and graph-based spatial memory for indoor-outdoor exploration.

## Launch Docker
```bash
cd ~/workspaces/isaac_ros-dev/
docker run --rm -it \
  --name nvblox_container \
  --net=host --ipc=host --privileged \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e ROS_DOMAIN_ID=0 \
  -e ROS_LOCALHOST_ONLY=0 \
  -e ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET \
  -e CYCLONEDDS_URI=file:///etc/cyclonedds/cyclonedds.xml \
  -e DDS_INTERFACE=enp2s0 \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e HF_HOME=/root/.cache/huggingface \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /dev:/dev \
  -v "$PWD/src":/workspaces/isaac_ros-dev/src:rw \
  -v "$PWD/build":/workspaces/isaac_ros-dev/build:rw \
  -v "$PWD/install":/workspaces/isaac_ros-dev/install:rw \
  -v "$PWD/log":/workspaces/isaac_ros-dev/log:rw \
  -v "$PWD/README.md":/workspaces/isaac_ros-dev/README.md:rw \
  -v "$PWD/hf_cache:/root/.cache/huggingface:rw \
  -v "$PWD/hf_home:/root/.huggingface:rw \
  --workdir /workspaces/isaac_ros-dev \
  graphnav-gpt:amd  \
  bash
```