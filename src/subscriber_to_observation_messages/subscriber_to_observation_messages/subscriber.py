import math
import time
import rclpy
import transforms3d
from rclpy.node import Node
from sensor_msgs.msg import JointState, LaserScan, Imu
from geometry_msgs.msg import PoseStamped
from diablo_joint_observer.msg import Observation
from tf_transformations import euler_from_quaternion

BASE_LINK_HEIGHT_OFFSET = 0.4925  # base_footprint -> base_link fixed joint z offset
# Freshness uses WALL time: sim time freezes during reset pauses, so old
# sensor data would otherwise still look fresh right after a respawn.
SENSOR_FRESH_WINDOW = 0.15        # seconds
# Additionally, a sensor message whose header stamp lags the current sim time
# by more than this is stale CONTENT (queued before a reset) and is rejected.
SENSOR_STAMP_MAX_LAG = 0.1        # seconds of sim time

class DiabloObserver(Node):
    def __init__(self):
        super().__init__('subscriber')

        # Publisher (observations)
        self.diablo_state_publisher = self.create_publisher(
            Observation,
            'observations',
            10
        )

        # Subscriber (/joint_states)
        self.joint_states_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        self.lidar_subscriber = self.create_subscription(
            LaserScan,
            'lidar',
            self.lidar_callback,
            10
        )   
        self.latest_lidar_ranges = []

        self.imu_subscriber = self.create_subscription(
            Imu,
            'imu',
            self.imu_callback,
            10 
        )
        self.latest_euler = []
        self.latest_acceleration = []

        # Model world pose (gz PosePublisher plugin) -> base_link height
        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            '/model/robot/pose',
            self.pose_callback,
            10
        )
        self.latest_height = None
        self.last_pose_time = None
        self.last_imu_time = None

    def stamp_is_current(self, stamp) -> bool:
        now = self.get_clock().now().nanoseconds * 1e-9
        if now <= 1e-6:
            return True  # sim time not started yet; cannot judge
        msg_time = stamp.sec + stamp.nanosec * 1e-9
        return (now - msg_time) <= SENSOR_STAMP_MAX_LAG

    def pose_callback(self, msg: PoseStamped):
        if not self.stamp_is_current(msg.header.stamp):
            return
        q = msg.pose.orientation
        q_norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        if not math.isfinite(q_norm) or q_norm < 0.5:
            return
        # z component of the model's body z-axis in world frame
        body_up = 1.0 - 2.0 * (q.x * q.x + q.y * q.y) / (q_norm * q_norm)
        height = msg.pose.position.z + BASE_LINK_HEIGHT_OFFSET * body_up
        if math.isfinite(height):
            self.latest_height = height
            self.last_pose_time = time.monotonic()

    def lidar_callback(self, msg: LaserScan):
        self.latest_lidar_ranges = list(msg.ranges)

    def imu_callback(self, msg: Imu):
        if not self.stamp_is_current(msg.header.stamp):
            return
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w

        q_norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if not math.isfinite(q_norm) or q_norm < 0.5:
            # degenerate/uninitialized orientation: not counted as fresh
            return

        roll, pitch, yaw = euler_from_quaternion([qx, qy, qz, qw])
        if not (math.isfinite(roll) and math.isfinite(pitch) and math.isfinite(yaw)):
            return
        self.latest_euler = [roll, pitch, yaw]

        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z

        self.latest_acceleration = [ax, ay, az]
        self.last_imu_time = time.monotonic()


    def joint_state_callback(self, msg: JointState):
        # Find indices of required joints
        try:
            joint_left_leg_1_index = msg.name.index("joint_left_leg_1")
            joint_right_leg_1_index = msg.name.index("joint_right_leg_1")   
            joint_left_leg_2_index = msg.name.index("joint_left_leg_2")
            joint_right_leg_2_index = msg.name.index("joint_right_leg_2")
            joint_left_leg_3_index = msg.name.index("joint_left_leg_3")
            joint_right_leg_3_index = msg.name.index("joint_right_leg_3")
            joint_left_leg_4_index = msg.name.index("joint_left_leg_4")
            joint_right_leg_4_index = msg.name.index("joint_right_leg_4")


        except ValueError:
            # Joint names are not in message yet
            return

        # Do not publish observations built from stale (pre-reset) sensor data
        if not self.sensors_are_fresh():
            return

        # Create message
        diablo_observation = Observation()
    

        #position of legs
        diablo_observation.left_leg_1_pos = msg.position[joint_left_leg_1_index]
        diablo_observation.right_leg_1_pos = msg.position[joint_right_leg_1_index]
        diablo_observation.left_leg_2_pos = msg.position[joint_left_leg_2_index]
        diablo_observation.right_leg_2_pos = msg.position[joint_right_leg_2_index]
        diablo_observation.left_leg_3_pos = msg.position[joint_left_leg_3_index]
        diablo_observation.right_leg_3_pos = msg.position[joint_right_leg_3_index]
        diablo_observation.left_leg_4_pos = msg.position[joint_left_leg_4_index]
        diablo_observation.right_leg_4_pos = msg.position[joint_right_leg_4_index]    

        #velocity of legs
        diablo_observation.left_leg_1_vel = msg.velocity[joint_left_leg_1_index]
        diablo_observation.right_leg_1_vel = msg.velocity[joint_right_leg_1_index]
        diablo_observation.left_leg_2_vel = msg.velocity[joint_left_leg_2_index]
        diablo_observation.right_leg_2_vel = msg.velocity[joint_right_leg_2_index]
        diablo_observation.left_leg_3_vel = msg.velocity[joint_left_leg_3_index]
        diablo_observation.right_leg_3_vel = msg.velocity[joint_right_leg_3_index]
        diablo_observation.left_leg_4_vel = msg.velocity[joint_left_leg_4_index]
        diablo_observation.right_leg_4_vel = msg.velocity[joint_right_leg_4_index]

        if self.latest_lidar_ranges:
            diablo_observation.lidar_ranges = self.latest_lidar_ranges

        if self.latest_euler:
            diablo_observation.imu_orientation = self.latest_euler

        if self.latest_acceleration:
            diablo_observation.acceleration = self.latest_acceleration

        diablo_observation.height = self.get_height()

        # Publish
        self.diablo_state_publisher.publish(diablo_observation)

    def sensors_are_fresh(self) -> bool:
        if self.last_pose_time is None or self.last_imu_time is None:
            return False
        now = time.monotonic()
        pose_age = now - self.last_pose_time
        imu_age = now - self.last_imu_time
        return pose_age <= SENSOR_FRESH_WINDOW and imu_age <= SENSOR_FRESH_WINDOW

    def get_height(self) -> float:
        if self.latest_height is None:
            return 0.0
        return float(self.latest_height)


def main(args=None):
    rclpy.init(args=args)
    node = DiabloObserver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
