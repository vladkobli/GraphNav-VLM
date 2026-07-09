#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import open3d as o3d


class DepthToPointCloudNode(Node):
    def __init__(self):
        super().__init__('depth_to_pointcloud')

        self.bridge = CvBridge()
        self.camera_info = None

        self.depth_sub = self.create_subscription(
            Image,
            '/clipseg/masked_depth',
            self.depth_callback,
            10
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera0/driver/aligned_depth_to_color/camera_info',
            self.info_callback,
            10
        )

        self.cloud_pub = self.create_publisher(
            PointCloud2,
            '/custom/depth_points',
            10
        )

        # Tuning parameters
        self.voxel_size = 0.01     # meters
        self.depth_trunc = 5.0      # meters
        self.z_min_keep = 0.20      # remove floor below this
        self.z_max_keep = 2.20      # remove ceiling above this

        self.get_logger().info('Depth to PointCloud node started.')

    def info_callback(self, msg: CameraInfo):
        self.camera_info = msg

    def depth_callback(self, msg: Image):
        if self.camera_info is None:
            return

        # Convert ROS depth image to numpy
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        # ROS depth encodings
        if msg.encoding == '16UC1':
            # mm -> m for reasoning/debug, but Open3D below will use uint16 mm
            depth_mm = depth.astype(np.uint16)
        elif msg.encoding == '32FC1':
            # meters -> uint16 mm for Open3D
            depth_mm = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth_mm = np.clip(depth_mm * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        else:
            self.get_logger().warn(f'Unsupported depth encoding: {msg.encoding}')
            return

        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=depth_mm.shape[1],
            height=depth_mm.shape[0],
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy
        )

        depth_o3d = o3d.geometry.Image(depth_mm)

        # Build Open3D point cloud from depth
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d,
            intrinsic,
            depth_scale=1000.0,
            depth_trunc=self.depth_trunc,
            project_valid_depth_only=True
        )

        # Optional refinement: crop in Z to remove floor/ceiling
        pts = np.asarray(pcd.points)
        if pts.size == 0:
            return

        mask = (pts[:, 2] >= self.z_min_keep) & (pts[:, 2] <= self.z_max_keep)
        pts = pts[mask]

        refined = o3d.geometry.PointCloud()
        refined.points = o3d.utility.Vector3dVector(pts)

        # Optional downsampling
        refined = refined.voxel_down_sample(voxel_size=self.voxel_size)

        out_pts = np.asarray(refined.points, dtype=np.float32)
        if out_pts.size == 0:
            return

        header = msg.header
        header.frame_id = self.camera_info.header.frame_id

        cloud_msg = point_cloud2.create_cloud_xyz32(
            header,
            out_pts.tolist()
        )
        self.cloud_pub.publish(cloud_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DepthToPointCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()