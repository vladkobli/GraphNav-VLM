#!/usr/bin/env bash
set -e

# Source ROS + RealSense workspace
source /opt/ros/${ROS_DISTRO}/setup.bash

if [ -f /ros2_ws/install/setup.bash ]; then
  source /ros2_ws/install/setup.bash
fi

if [ -f /rgbd_camera_intel_dev/install/setup.bash ]; then
  source /rgbd_camera_intel_dev/install/setup.bash
fi

# Generate CycloneDDS config from template
mkdir -p /etc/cyclonedds

DDS_IFACE="${DDS_INTERFACE:-}"

if [ -z "$DDS_IFACE" ]; then
  echo "[entrypoint] DDS_INTERFACE not set. Trying to auto-detect default interface..."
  DDS_IFACE="$(ip route | awk '/default/ {print $5; exit}')"
fi

if [ -z "$DDS_IFACE" ]; then
  echo "[entrypoint] Could not detect DDS interface. Falling back to eth0."
  DDS_IFACE="eth0"
fi

echo "[entrypoint] Using DDS interface: ${DDS_IFACE}"

if [ -f /etc/cyclonedds/cyclonedds.template.xml ]; then
  sed "s/__DDS_INTERFACE__/${DDS_IFACE}/g" \
    /etc/cyclonedds/cyclonedds.template.xml \
    > /etc/cyclonedds/cyclonedds.xml
fi

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file:///etc/cyclonedds/cyclonedds.xml}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"
echo "[entrypoint] ROS_AUTOMATIC_DISCOVERY_RANGE=${ROS_AUTOMATIC_DISCOVERY_RANGE}"
echo "[entrypoint] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo "[entrypoint] CYCLONEDDS_URI=${CYCLONEDDS_URI}"
echo "[entrypoint] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"

exec "$@"