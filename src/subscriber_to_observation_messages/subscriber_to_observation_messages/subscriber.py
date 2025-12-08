import rclpy
import transforms3d
from rclpy.node import Node
from sensor_msgs.msg import JointState, LaserScan, Imu
from diablo_joint_observer.msg import Observation
from tf_transformations import euler_from_quaternion

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

    def lidar_callback(self, msg: LaserScan):
        self.latest_lidar_ranges = list(msg.ranges)

    def imu_callback(self, msg: Imu):
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w

        roll, pitch, yaw = euler_from_quaternion([qx, qy, qz, qw])
        self.latest_euler = [roll, pitch, yaw]

        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z

        self.latest_acceleration = [ax, ay, az]


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

        # Publish
        self.diablo_state_publisher.publish(diablo_observation)


def main(args=None):
    rclpy.init(args=args)
    node = DiabloObserver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
