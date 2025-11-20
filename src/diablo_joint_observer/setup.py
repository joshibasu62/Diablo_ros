from setuptools import find_packages, setup
import os
import glob

package_name = 'diablo_joint_observer'

def get_files(folder, pattern='*'):
    """Helper function to return list of files matching pattern in folder"""
    return glob.glob(os.path.join(folder, pattern))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Package resource
        ('share/ament_index/resource_index/packages', [os.path.join('resource', package_name)]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/msg', get_files('msg', '*.msg')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='basanta-joshi',
    maintainer_email='joshibasu62@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
