from setuptools import setup, find_packages
import glob
import os

package_name = 'diablo_env_ros'

def get_files(folder, pattern='*'):
    """Helper function to return list of files matching pattern in folder"""
    return glob.glob(os.path.join(folder, pattern))

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Package resource
        ('share/ament_index/resource_index/packages', [os.path.join('resource', package_name)]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', get_files('launch', '*.py')),
        ('share/' + package_name + '/urdf', get_files('urdf', '*.urdf') + 
                                            get_files('urdf', '*.xacro') + 
                                            get_files('urdf', '*.gazebo')),
        ('share/' + package_name + '/meshes', get_files('meshes', '*.STL')),
        ('share/' + package_name + '/rviz', get_files('rviz', '*.rviz')),
        ('share/' + package_name + '/config', get_files('config', '*.yaml')),
        ('share/' + package_name + '/worlds', get_files('worlds', '*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Basanta Joshi',
    maintainer_email='joshibasu62@gmail.com',
    description='Diablo ROS 2 environment',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Add ROS 2 Python nodes here, e.g.
            # 'diablo_node = diablo_env_ros.some_module:main'
        ],
    },
)
