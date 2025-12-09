from setuptools import find_packages, setup
from generate_parameter_library_py.setup_helper import generate_parameter_module

package_name = 'base_class'

generate_parameter_module(
    "diablo_reinforcement_learning_parameters",
    "base_class/reinforcement_learning_node_parameters.yaml"
)

# generate_parameter_module(
#     "Q_learning_base_parameters",
#     "base_class/Q_learning_base_parameters.yaml"
# )



setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            "venv_check = base_class.venv_check:main",
            "diablo_basic_policy_node = base_class.diablo_basic_policy_node:main",
            "reinforce = base_class.reinforce:main",
            "reinforce_node = base_class.reinforce_node:main",
            "actor_critic_node = base_class.actor_critic_node:main",
        ],
    },
)
