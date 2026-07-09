from setuptools import find_packages, setup
from glob import glob

package_name = 'sensors_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'python-dotenv'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='vladkoblicica1@gmail.com',
    description='AGRO-GPT is a ROS2 package that integrates the capabilities of ChatGPT with agricultural robotics, enabling intelligent navigation and decision-making for enhanced farming operations.',
    license='ROCON',
    entry_points={
        'console_scripts': [
        ],
    },
)
