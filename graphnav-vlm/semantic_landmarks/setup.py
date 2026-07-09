from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'semantic_landmarks'

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
            "clipseg_depth_node_with_multimasks = semantic_landmarks.clipseg_depth_node_with_multimasks:main",
            "masked_depth_landmark_logger = semantic_landmarks.masked_depth_landmark_logger:main",
            "clipseg_depth_node_with_multimasks_2 = semantic_landmarks.clipseg_depth_node_with_multimasks_2:main",
            "masked_depth_landmark_logger_2 = semantic_landmarks.masked_depth_landmark_logger_2:main",
        ],
    },
)
