from rclpy.node import Node
from diablo_joint_observer.msg import Observation
from rclpy.subscription import Subscription
from std_msgs.msg import Float64MultiArray, Float64
from std_srvs.srv import Empty
from rclpy.task import Future
from rclpy.publisher import Publisher
from rclpy.client import Client
import math
import time

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
         # 8 joint positions + 8 joint velocities + 1 base_link height + 2 imu (roll, pitch) + 1 vertical accel = 20
        self.imu_data = []
        self.lidar_data = []
        self.acceleration_data = []
        self.is_truncated: bool = False
        self.height_limit_lower: float = 0.15
        self.height_limit_upper: float = 0.75
        self.max_tilt: float = math.radians(45.0)
        # Here i will add limit later using inverse kinematics i.e distace of baselink from ground while writing reinforcement learning code
        self.restarting_future: Future = None
        self.is_resetting: bool = False
        self.has_fresh_observation: bool = False
        self.reset_finished_at: float = None

    # After the respawn service returns, keep dropping observations for this
    # long (wall time). The observer's own freshness window (0.15 s, wall
    # time) then guarantees that any observation accepted afterwards was built
    # from sensors of the NEW robot, not the pre-remove one.
    RESET_SETTLE_SECONDS = 0.2

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

        ranges = diablo_observation.lidar_ranges
        self.lidar_data = ranges if len(ranges) >= 3 else [0.0, 0.0, 0.0]
        self.diablo_observations[16] = diablo_observation.height

        imu_orientation = diablo_observation.imu_orientation
        if len(imu_orientation) >= 3:
            self.imu_data = imu_orientation
            self.diablo_observations[17] = self.imu_data[0]  # roll
            self.diablo_observations[18] = self.imu_data[1]  # pitch

        acceleration = diablo_observation.acceleration
        if len(acceleration) >= 3:
            self.acceleration_data = acceleration
            self.diablo_observations[19] = self.acceleration_data[2]  # az

        # self.diablo_observations[19] = self.imu_data[2]  # yaw

        self.has_fresh_observation = True
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
        self.has_fresh_observation = False

    def update_simulation_status(self):
        if self.is_truncated:
            return

        height = self.diablo_observations[16]
        roll = self.diablo_observations[17]
        pitch = self.diablo_observations[18]

        reason = None
        if not math.isfinite(height) or not math.isfinite(roll) or not math.isfinite(pitch):
            reason = f"non-finite obs (height={height}, roll={roll}, pitch={pitch})"
        elif height < self.height_limit_lower or height > self.height_limit_upper:
            reason = f"height {height:.3f} outside [{self.height_limit_lower}, {self.height_limit_upper}]"
        elif abs(roll) > self.max_tilt or abs(pitch) > self.max_tilt:
            reason = f"tilt roll={roll:.3f} pitch={pitch:.3f}"

        if reason is not None:
            self.is_truncated = True
            self.get_logger().info(f"Truncated after {self.step} steps: {reason}")


    def restart_simulation(self):
        while not self.simulation_reset_service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("restart_sim_service not available, waiting again...")
        self.restarting_future = self.simulation_reset_service_client.call_async(Empty.Request())

    def is_simulation_ready(self) -> bool:
        if self.restarting_future is not None:
            if not self.restarting_future.done():
                return False
            if self.is_resetting:
                if self.reset_finished_at is None:
                    self.reset_finished_at = time.monotonic()
                    return False
                if time.monotonic() - self.reset_finished_at < self.RESET_SETTLE_SECONDS:
                    return False
                self.is_resetting = False
        return self.has_fresh_observation

    def restart_learning_loop(self):
        self.is_resetting = True
        self.reset_finished_at = None
        self.restart_simulation()
        self.reset_observation()
        time.sleep(0.2)



