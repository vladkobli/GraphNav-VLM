from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robot_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='vladkoblicica1@gmail.com',
    description='Bridge PointStamped target points to Nav2 NavigateToPose goals',
    license='Apache-2.0',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'clipseg_depth_node_with_mask = robot_navigation.clipseg_depth_node_with_mask:main',
            'clipseg_target_point_node = robot_navigation.clipseg_target_point_node:main',
            'clipseg_depth_node_navigation = robot_navigation.clipseg_depth_node_navigation:main',
            'crop_lidar = robot_navigation.crop_lidar:main',
            'rgbd2pointcloud = robot_navigation.rgbd2pointcloud:main',
            'pointcloud_height_filter = robot_navigation.pointcloud_height_filter:main',
            'target_to_nav2 = robot_navigation.target_to_nav2:main',
            'target_to_nav2_once = robot_navigation.target_to_nav2_once:main',
            'masked_depth_target_point = robot_navigation.masked_depth_target_point:main',
            'twist_to_twist_stamped = robot_navigation.twist_to_twist_stamped:main',
            'resize_camera_info_node = robot_navigation.resize_camera_info_node:main',
            'resize_image_node = robot_navigation.resize_image_node:main',
            'panther_frontier_nav2_client = robot_navigation.panther_frontier_nav2_client:main',
            'panther_frontier_marker_probe = robot_navigation.panther_frontier_marker_probe:main',
            'startup_360_scan = robot_navigation.startup_360_scan:main',
        ],
    },
)
