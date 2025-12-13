#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/entity.pb.h>
#include <gz/msgs/entity_factory.pb.h>
#include <chrono>
#include <gz/transport/Node.hh>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/empty.hpp>

std::optional<std::string> get_robot_description_from_topic()
{
  const std::string topic_name{"robot_description"};
  std::promise<std::string> robot_description_promise;
  std::shared_future<std::string> robot_description_future(robot_description_promise.get_future());
  rclcpp::executors::SingleThreadedExecutor executor;
  auto ros2_node = std::make_shared<rclcpp::Node>("robot_description_acquire_node");
  executor.add_node(ros2_node);
  const auto description_subs = ros2_node->create_subscription<std_msgs::msg::String>(
    topic_name, rclcpp::QoS(1).transient_local(),
    [&robot_description_promise](const std_msgs::msg::String::SharedPtr msg)
    { robot_description_promise.set_value(msg->data); });

  rclcpp::FutureReturnCode future_ret;
  while (rclcpp::ok() && future_ret != rclcpp::FutureReturnCode::SUCCESS)
  {
    RCLCPP_INFO(ros2_node->get_logger(), "Waiting messages on topic [%s].", topic_name.c_str());
    future_ret = executor.spin_until_future_complete(robot_description_future, std::chrono::seconds(1));
  }

  if (future_ret != rclcpp::FutureReturnCode::SUCCESS)
  {
    RCLCPP_ERROR(ros2_node->get_logger(), "Failed to get XML from topic [%s].", topic_name.c_str());
    return std::nullopt;
  }
  return robot_description_future.get();
}

class SimulationControlNode : public rclcpp::Node
{
public:
  SimulationControlNode(const rclcpp::NodeOptions& options) : rclcpp::Node("simulation_control_node", options)
  {
    robot_description_ = *get_robot_description_from_topic();
    server_ = create_service<std_srvs::srv::Empty>(
      "restart_sim_service",
      [&](std_srvs::srv::Empty::Request::SharedPtr, std_srvs::srv::Empty::Response::SharedPtr)
      {
        execute_gazebo_request(build_remove_request(), service_remove_);
        robot_name_ = std::string("diablo") + std::to_string(++counter_);
        execute_gazebo_request(build_create_request(), service_create_);
        rclcpp::sleep_for(
          std::chrono::milliseconds(500));  // Spawning model is unpredictable so arbitrary delay is used
      });
  }

  gz::msgs::Entity build_remove_request() const
  {
    gz::msgs::Entity robot_remove_request;
    robot_remove_request.set_name(robot_name_);
    robot_remove_request.set_type(gz::msgs::Entity_Type_MODEL);

    return robot_remove_request;
  }

  gz::msgs::EntityFactory build_create_request() const
  {
    gz::msgs::EntityFactory robot_spawn_request;
    robot_spawn_request.set_sdf(robot_description_);
    robot_spawn_request.set_name(robot_name_);

    return robot_spawn_request;
  }

  template <typename T>
  void execute_gazebo_request(T request, const std::string& service_name)
  {
    gz::msgs::Boolean response;
    bool result;
    const unsigned int timeout{5000};
    while (rclcpp::ok() and not node_.Request(service_name, request, timeout, response, result))
    {
      RCLCPP_WARN(this->get_logger(), "Waiting for service [%s] to become available ...", service_name.c_str());
    }
  }

  gz::transport::Node node_{};
  int counter_{};
  std::string robot_name_{"robot"};
  std::string robot_description_{};
  const std::string service_create_{"/world/empty/create"};
  const std::string service_remove_{"/world/empty/remove"};
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr server_;
};

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(SimulationControlNode)






// #include <chrono>
// #include <optional>
// #include <string>

// #include <rclcpp/rclcpp.hpp>
// #include <std_srvs/srv/empty.hpp>

// #include <gz/transport/Node.hh>
// #include <gz/msgs/boolean.pb.h>
// #include <gz/msgs/world_control.pb.h>
// #include <gz/msgs/pose.pb.h>
// #include <gz/msgs/entity.pb.h>

// class SimulationControlNode : public rclcpp::Node
// {
// public:
//   SimulationControlNode(const rclcpp::NodeOptions & options)
//   : rclcpp::Node("simulation_control_node", options)
//   {
//     robot_name_ = "diablo";   // MUST stay constant

//     server_ = create_service<std_srvs::srv::Empty>(
//       "restart_sim_service",
//       std::bind(
//         &SimulationControlNode::restart_simulation,
//         this,
//         std::placeholders::_1,
//         std::placeholders::_2
//       )
//     );

//     RCLCPP_INFO(get_logger(), "SimulationControlNode ready (pose reset mode)");
//   }

// private:
//   // ------------------- Service callback -------------------

//   void restart_simulation(
//     const std_srvs::srv::Empty::Request::SharedPtr,
//     std_srvs::srv::Empty::Response::SharedPtr)
//   {
//     RCLCPP_INFO(get_logger(), "Resetting simulation (pose + velocity)");

//     pause_physics(true);

//     reset_robot_velocity();
//     reset_robot_pose();

//     rclcpp::sleep_for(std::chrono::milliseconds(50));

//     pause_physics(false);
//   }

//   // ------------------- Gazebo helpers -------------------

//   void pause_physics(bool pause)
//   {
//     gz::msgs::WorldControl msg;
//     msg.set_pause(pause);

//     gz::msgs::Boolean response;
//     bool result{false};

//     node_.Request(
//       "/world/empty/control",
//       msg,
//       1000,
//       response,
//       result
//     );
//   }

//   void reset_robot_pose()
//   {
//     gz::msgs::Pose pose_msg;
//     pose_msg.set_name(robot_name_);

//     // Position (adjust Z for your robot)
//     pose_msg.mutable_position()->set_x(0.0);
//     pose_msg.mutable_position()->set_y(0.0);
//     pose_msg.mutable_position()->set_z(0.35);

//     // Orientation (identity quaternion)
//     pose_msg.mutable_orientation()->set_w(1.0);
//     pose_msg.mutable_orientation()->set_x(0.0);
//     pose_msg.mutable_orientation()->set_y(0.0);
//     pose_msg.mutable_orientation()->set_z(0.0);

//     gz::msgs::Boolean response;
//     bool result{false};

//     node_.Request(
//       "/world/empty/set_pose",
//       pose_msg,
//       1000,
//       response,
//       result
//     );
//   }

//   void reset_robot_velocity()
//   {
//     gz::msgs::Entity entity_msg;
//     entity_msg.set_name(robot_name_);
//     entity_msg.set_type(gz::msgs::Entity_Type_MODEL);

//     gz::msgs::Boolean response;
//     bool result{false};

//     node_.Request(
//       "/world/empty/reset_entity",
//       entity_msg,
//       1000,
//       response,
//       result
//     );
//   }

//   // ------------------- Members -------------------

//   gz::transport::Node node_;
//   std::string robot_name_;
//   rclcpp::Service<std_srvs::srv::Empty>::SharedPtr server_;
// };

// #include "rclcpp_components/register_node_macro.hpp"
// RCLCPP_COMPONENTS_REGISTER_NODE(SimulationControlNode)

