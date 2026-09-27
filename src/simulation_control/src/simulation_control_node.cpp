
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/empty.hpp>

#include <gz/transport/Node.hh>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/world_control.pb.h>
#include <gz/msgs/entity.pb.h>
#include <gz/msgs/entity_factory.pb.h>

class SimulationControlNode : public rclcpp::Node
{
public:
  SimulationControlNode(const rclcpp::NodeOptions & options)
  : rclcpp::Node("simulation_control_node", options)
  {
    robot_name_ = "robot";   // MUST stay constant (matches spawn name in launch)
    world_name_ = "empty";

    spawn_x_ = declare_parameter<double>("spawn_x", 0.0);
    spawn_y_ = declare_parameter<double>("spawn_y", 0.0);
    spawn_z_ = declare_parameter<double>("spawn_z", 0.0);

    std::string robot_description_path =
      declare_parameter<std::string>("robot_description_path", "");

    if (!loadRobotDescription(robot_description_path)) {
      RCLCPP_ERROR(
        get_logger(),
        "Could not load robot description from '%s'; restart_sim_service will fail until fixed",
        robot_description_path.c_str());
    }

    control_service_ = "/world/" + world_name_ + "/control";
    remove_service_ = "/world/" + world_name_ + "/remove/blocking";
    create_service_ = "/world/" + world_name_ + "/create/blocking";

    server_ = create_service<std_srvs::srv::Empty>(
      "restart_sim_service",
      std::bind(
        &SimulationControlNode::restart_simulation,
        this,
        std::placeholders::_1,
        std::placeholders::_2
      )
    );

    // Self-heal: a previous session may have left physics paused (e.g. this
    // node was killed mid-reset). Paused physics freezes /clock, which
    // freezes every sim-time timer in the ROS graph.
    if (!pause_physics(false)) {
      RCLCPP_WARN(
        get_logger(),
        "Could not unpause physics at startup (gz world control not available yet?)");
    }

    RCLCPP_INFO(get_logger(), "SimulationControlNode ready (remove + recreate reset mode)");
  }

private:
  // ------------------- Service callback -------------------

  void restart_simulation(
    const std_srvs::srv::Empty::Request::SharedPtr,
    std_srvs::srv::Empty::Response::SharedPtr)
  {
    if (robot_description_.empty()) {
      RCLCPP_ERROR(get_logger(), "Robot description not loaded, cannot reset");
      return;
    }

    RCLCPP_INFO(get_logger(), "Resetting simulation (remove + recreate)");

    pause_physics(true);

    remove_robot();
    create_robot();

    rclcpp::sleep_for(std::chrono::milliseconds(50));

    if (!pause_physics(false)) {
      RCLCPP_ERROR(
        get_logger(),
        "Failed to unpause physics after reset - simulation will be frozen!");
    }
  }

  // ------------------- Robot description -------------------

  bool loadRobotDescription(const std::string & path)
  {
    std::string resolved_path = path;
    if (resolved_path.empty()) {
      resolved_path = findDefaultRobotDescription();
    }

    if (resolved_path.empty()) {
      return false;
    }

    std::ifstream file(resolved_path);
    if (!file.is_open()) {
      RCLCPP_ERROR(get_logger(), "Failed to open robot description file: %s", resolved_path.c_str());
      return false;
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();
    robot_description_ = buffer.str();

    if (robot_description_.empty()) {
      RCLCPP_ERROR(get_logger(), "Robot description file is empty: %s", resolved_path.c_str());
      return false;
    }

    RCLCPP_INFO(get_logger(), "Loaded robot description from %s", resolved_path.c_str());
    return true;
  }

  std::string findDefaultRobotDescription()
  {
    const char * ament_prefix_path = std::getenv("AMENT_PREFIX_PATH");
    if (ament_prefix_path == nullptr) {
      return "";
    }

    const std::string relative_path = "/share/diablo_env_ros/urdf/robot.urdf";
    std::stringstream stream(ament_prefix_path);
    std::string prefix;
    while (std::getline(stream, prefix, ':')) {
      if (prefix.empty()) {
        continue;
      }
      std::string candidate = prefix + relative_path;
      std::ifstream file(candidate);
      if (file.is_open()) {
        return candidate;
      }
    }
    return "";
  }

  // ------------------- Gazebo helpers -------------------

  bool pause_physics(bool pause)
  {
    gz::msgs::WorldControl msg;
    msg.set_pause(pause);

    gz::msgs::Boolean response;
    bool result{false};

    node_.Request(
      control_service_,
      msg,
      1000,
      response,
      result
    );
    return result;
  }

  void remove_robot()
  {
    gz::msgs::Entity msg;
    msg.set_name(robot_name_);
    msg.set_type(gz::msgs::Entity_Type_MODEL);

    gz::msgs::Boolean response;
    bool result{false};

    if (!node_.Request(remove_service_, msg, 3000, response, result)) {
      RCLCPP_WARN(get_logger(), "remove service request to %s failed", remove_service_.c_str());
      return;
    }
    if (!response.data()) {
      RCLCPP_WARN(get_logger(), "remove of model '%s' reported failure", robot_name_.c_str());
    }
  }

  void create_robot()
  {
    gz::msgs::EntityFactory msg;
    msg.set_name(robot_name_);
    msg.set_allow_renaming(false);
    msg.set_sdf(robot_description_);

    auto * pose = msg.mutable_pose();
    pose->mutable_position()->set_x(spawn_x_);
    pose->mutable_position()->set_y(spawn_y_);
    pose->mutable_position()->set_z(spawn_z_);
    pose->mutable_orientation()->set_w(1.0);
    pose->mutable_orientation()->set_x(0.0);
    pose->mutable_orientation()->set_y(0.0);
    pose->mutable_orientation()->set_z(0.0);

    gz::msgs::Boolean response;
    bool result{false};

    if (!node_.Request(create_service_, msg, 5000, response, result)) {
      RCLCPP_ERROR(get_logger(), "create service request to %s failed", create_service_.c_str());
      return;
    }
    if (!response.data()) {
      RCLCPP_ERROR(get_logger(), "create of model '%s' reported failure", robot_name_.c_str());
    }
  }

  // ------------------- Members -------------------

  gz::transport::Node node_;
  std::string robot_name_;
  std::string world_name_;
  std::string robot_description_;
  std::string control_service_;
  std::string remove_service_;
  std::string create_service_;
  double spawn_x_{0.0};
  double spawn_y_{0.0};
  double spawn_z_{0.0};
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr server_;
};

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(SimulationControlNode)
