from rclpy.node import Node
from diablo_joint_observer.msg import Observation
from rclpy.subscription import Subscription
from std_msgs.msg import Float64MultiArray, Float64
from std_srvs.srv import Empty
from rclpy.task import Future
from rclpy.publisher import Publisher
from rclpy.client import Client

class DiabloBaseNode(Node):
    def __init__(self, node_name = "base_node_class"):
        super().__init__(node_name)
        self.observation_subscriber : Subscription = self.create_subscription(
            Observation,
            'observations',
            self.store_observation,
            10
        )

        self.simulation_reset_service_client: Client = self.create_client(Empty, "restart_sim_service")
        self.effort_command_publisher: Publisher = self.create_publisher(Float64MultiArray, "effort_cmd", 10)
        self.diablo_observations: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.lidar_data = []
        self.is_truncated: bool = False
        # self.UPPER_Link_limit: float = 1.4
        # self.LOWER_Link_limit: float = -1.4
        self.height_limit: float = 0.25
        # Here i will add limit later using inverse kinematics i.e distace of baselink from ground while writing reinforcement learning code
        self.restarting_future: Future = None
        self.is_resetting: bool = False

    def store_observation(self, diablo_observation: Observation):  
        if self.is_resetting:
            return
         
        self.diablo_observations[0] = diablo_observation.left_leg_1_pos
        self.diablo_observations[1] = diablo_observation.right_leg_1_pos
        self.diablo_observations[2] = diablo_observation.left_leg_2_pos
        self.diablo_observations[3] = diablo_observation.right_leg_2_pos
        self.diablo_observations[4] = diablo_observation.left_leg_3_pos
        self.diablo_observations[5] = diablo_observation.right_leg_3_pos
        self.diablo_observations[6] = diablo_observation.left_leg_4_pos
        self.diablo_observations[7] = diablo_observation.right_leg_4_pos

        self.diablo_observations[8] = diablo_observation.left_leg_1_vel
        self.diablo_observations[9] = diablo_observation.right_leg_1_vel
        self.diablo_observations[10] = diablo_observation.left_leg_2_vel
        self.diablo_observations[11] = diablo_observation.right_leg_2_vel
        self.diablo_observations[12] = diablo_observation.left_leg_3_vel
        self.diablo_observations[13] = diablo_observation.right_leg_3_vel
        self.diablo_observations[14] = diablo_observation.left_leg_4_vel
        self.diablo_observations[15] = diablo_observation.right_leg_4_vel
        self.lidar_data = diablo_observation.lidar_ranges
        self.diablo_observations[16] = self.lidar_data[0]
        self.diablo_observations[17] = self.lidar_data[1]
        self.diablo_observations[18] = self.lidar_data[2]
        

        self.update_simulation_status()

    def get_diablo_observations(self) -> list[float]:
        return self.diablo_observations
    
    def is_simulation_stopped(self) -> bool:
        return self.is_truncated

    def take_action(self, action: Float64MultiArray):
        self.effort_command_publisher.publish(action)

    def reset_observation(self):
        self.diablo_observations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.lidar_data = []
        self.is_truncated = False

    def update_simulation_status(self):

        current_distance = self.diablo_observations[17]
        self.get_logger().info(f"Ground distance: {current_distance}, Truncated: {self.is_truncated}")
        
        if self.diablo_observations[17] < self.height_limit:
            self.is_truncated = True
    

    def restart_simulation(self):
        while not self.simulation_reset_service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("restart_sim_service not available, waiting again...")
        self.restarting_future = self.simulation_reset_service_client.call_async(Empty.Request())

    def is_simulation_ready(self) -> bool:
        if self.restarting_future is None:
            return True
        try:
            if self.restarting_future.done():
                self.is_resetting = False
                return True
            return False
        except:
            return False

    def restart_learning_loop(self):
        self.is_resetting = True
        self.restart_simulation()
        self.reset_observation()



