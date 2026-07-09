from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'graph_nav'

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
    description='Graph-based semantic navigation for ROS2 using VLMs and LLMs',
    license='Apache-2.0',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'graph_nav_state_machine = graph_nav.state_machine:main', # state_machine works with add_description2.py
            'graph_nav_state_machine2 = graph_nav.state_machine2:main', # state_machine2 works with add_description.py
            'graph_nav_state_machine3 = graph_nav.state_machine3:main', # state_machine3 works with add_description.py
            'graph_nav_moondream_server = graph_nav.add_description:main',
            'graph_nav_moondream_server2 = graph_nav.add_description2:main',
        ],
    },
)
