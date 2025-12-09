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
        self.effort_command_publisher: Publisher = [
                                                    self.create_publisher(Float64, 'joint_left_leg_1_effort', 10),
                                                    self.create_publisher(Float64, 'joint_right_leg_1_effort', 10),
                                                    self.create_publisher(Float64, 'joint_left_leg_2_effort', 10),
                                                    self.create_publisher(Float64, 'joint_right_leg_2_effort', 10),
                                                    self.create_publisher(Float64, 'joint_left_leg_3_effort', 10),
                                                    self.create_publisher(Float64, 'joint_right_leg_3_effort', 10),
                                                    self.create_publisher(Float64, 'joint_left_leg_4_effort', 10),
                                                    self.create_publisher(Float64, 'joint_right_leg_4_effort', 10),
                                                ]
        self.diablo_observations: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
         # 8 joint positions + 8 joint velocities + 1 lidar distance + 3 imu orientations = 20
        self.imu_data = []
        self.lidar_data = []
        self.acceleration_data = []
        self.is_truncated: bool = False
        self.height_limit_lower: float = 0.15
        self.height_limit_upper: float = 0.75
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
        self.diablo_observations[16] = self.lidar_data[1]

        self.imu_data = diablo_observation.imu_orientation
        self.diablo_observations[17] = self.imu_data[0]  # roll
        self.diablo_observations[18] = self.imu_data[1]  # pitch

        self.acceleration_data = diablo_observation.acceleration
        self.diablo_observations[19] = self.acceleration_data[2]  # az
        
        # self.diablo_observations[19] = self.imu_data[2]  # yaw
        
        

        self.update_simulation_status()

    def get_diablo_observations(self) -> list[float]:
        return self.diablo_observations
    
    def is_simulation_stopped(self) -> bool:
        return self.is_truncated

    def take_action(self, action_list):
        for i, publisher in enumerate(self.effort_command_publisher):
            msg = Float64()
            msg.data = action_list[i]
            publisher.publish(msg)

    def reset_observation(self):
        self.diablo_observations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.lidar_data = [0.0, 0.0, 0.0]
        self.imu_data = [0.0, 0.0, 0.0]
        self.acceleration_data = [0.0, 0.0, 0.0]
        self.is_truncated = False

    def update_simulation_status(self):

        # current_distance = self.diablo_observations[16]
        # self.get_logger().info(f"Ground distance: {current_distance}, Truncated: {self.is_truncated}")
        
        # if self.diablo_observations[16] < self.height_limit_lower or self.diablo_observations[16] > self.height_limit_upper:
        #     self.is_truncated = True

        if self.diablo_observations[16] < 0:
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



