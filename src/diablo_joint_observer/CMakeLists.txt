cmake_minimum_required(VERSION 3.8)
project(diablo_joint_observer)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Observation.msg"
  DEPENDENCIES std_msgs
)

ament_package()
