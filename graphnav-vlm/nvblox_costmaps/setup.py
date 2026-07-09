from setuptools import find_packages, setup

package_name = 'nvblox_costmaps'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/realsense_nvblox.launch.py']),
        ('share/' + package_name + '/launch', ['launch/realsense_velodyne_nvblox.launch.py']),
        ('share/' + package_name + '/launch', ['launch/velodyne_nvblox.launch.py']),
        ('share/' + package_name + '/launch', ['launch/autostart_realsense_velodyne_nvblox.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vladkobli',
    maintainer_email='vladkoblicica1@gmail.com',
    description='NvBlox costmap generation using RealSense and/or Velodyne sensors',
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'nvblox_depth_relay = nvblox_costmaps.depth_relay:main',
        ],
    },
)
