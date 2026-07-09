from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'camera_capture'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='vladkoblicica1@gmail.com',
    description='Bridge PointStamped target points to Nav2 NavigateToPose goals',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "camera_capture = camera_capture.camera_capture:main",
            "waypoint_dataset_logger = camera_capture.waypoint_dataset_logger:main",
            "manual_pose_sequence_logger = camera_capture.manual_pose_sequence_logger:main",
            "lerobot_realsense_sequence_logger = camera_capture.lerobot_realsense_sequence_logger:main",
            "lerobot_camera_tf_broadcaster = camera_capture.lerobot_camera_tf_broadcaster:main",
            "lerobot_camera_sweep_server = camera_capture.lerobot_camera_sweep_server:main",
            "show_map_nodes = camera_capture.show_map_nodes:main",
            "build_dataset_graph = camera_capture.build_dataset_graph:main",
        ],
    },
)
