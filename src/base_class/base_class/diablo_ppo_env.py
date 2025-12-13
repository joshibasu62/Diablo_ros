# diablo_ppo_env.py

import time
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from rclpy.task import Future

from std_msgs.msg import Float64
from std_srvs.srv import Empty
from diablo_joint_observer.msg import Observation


class DiabloRosNode(Node):
    """Internal ROS2 node used by the Gym environment."""

    def __init__(self, max_effort_command):
        super().__init__("diablo_ppo_env_node")

        # Store command limits (8 joints)
        self.max_effort_command = np.array(max_effort_command, dtype=np.float32)

        # Latest observation buffer
        self.latest_obs: Optional[Observation] = None
        self.obs_seq: int = 0  # increment on each new obs

        # Simulation status
        self.is_truncated: bool = False

        # Subscribers
        self.observation_sub = self.create_subscription(
            Observation,
            "observations",
            self._obs_callback,
            10,
        )

        # Publishers for 8 joint efforts
        joint_topics = [
            "joint_left_leg_1_effort",
            "joint_right_leg_1_effort",
            "joint_left_leg_2_effort",
            "joint_right_leg_2_effort",
            "joint_left_leg_3_effort",
            "joint_right_leg_3_effort",
            "joint_left_leg_4_effort",
            "joint_right_leg_4_effort",
        ]
        self.effort_pubs = [
            self.create_publisher(Float64, topic, 10) for topic in joint_topics
        ]

        # Reset service client
        self.reset_client = self.create_client(Empty, "restart_sim_service")

    # ------------------------------------------------------------------ #
    # Callbacks & helpers                                                #
    # ------------------------------------------------------------------ #

    def _obs_callback(self, msg: Observation):
        self.latest_obs = msg
        self.obs_seq += 1
        # Example truncation criterion using lidar distance:
        # Use the same logic as your DiabloBaseNode.update_simulation_status()
        if len(msg.lidar_ranges) > 1 and msg.lidar_ranges[1] < 0.0:
            self.is_truncated = True

    def publish_torques(self, torques: np.ndarray):
        """Publish 8 torques to Gazebo."""
        for i, pub in enumerate(self.effort_pubs):
            msg = Float64()
            msg.data = float(torques[i])
            pub.publish(msg)

    def request_reset(self):
        """Call the restart_sim_service and wait for completion."""
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for restart_sim_service...")

        future: Future = self.reset_client.call_async(Empty.Request())
        # Block (from caller's point of view) until reset is done
        while not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)


class DiabloEnv(gym.Env):
    """Gymnasium environment wrapping your ROS2+Gazebo Diablo simulation."""

    metadata = {"render_modes": []}

    def __init__(self,
                 max_effort_command,
                 max_steps: int = 2096,
                 height_limit_lower: float = 0.15,
                 height_limit_upper: float = 0.75):
        super().__init__()

        self.max_steps = max_steps
        self.height_limit_lower = height_limit_lower
        self.height_limit_upper = height_limit_upper

        # Observation: 20 floats (same as your DiabloBaseNode)
        self.obs_dim = 20
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # Action: 8D continuous in [-1, 1]
        self.act_dim = len(max_effort_command)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.act_dim,),
            dtype=np.float32
        )

        # ROS2 node
        self.node = DiabloRosNode(max_effort_command=max_effort_command)

        # Internal state
        self.current_obs: Optional[np.ndarray] = None
        self.step_count: int = 0

        # For detecting new observation per step
        self._last_obs_seq: int = -1

    # ------------------------------------------------------------------ #
    # Gym API                                                            #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Request simulation reset
        self.node.is_truncated = False
        self.node.request_reset()

        # Clear internal counters
        self.step_count = 0
        self._last_obs_seq = -1
        self.current_obs = None

        # Wait for the first fresh observation
        obs = self._wait_for_new_observation(timeout_sec=5.0)
        if obs is None:
            raise RuntimeError("No observation received after reset")

        self.current_obs = obs
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        # Clip and scale action to torques
        action = np.clip(action, self.action_space.low, self.action_space.high)
        torques = action * self.node.max_effort_command
        self.node.publish_torques(torques)

        # Wait for next observation from ROS
        obs = self._wait_for_new_observation(timeout_sec=0.5)
        if obs is None:
            # No new obs: treat as truncated to be safe
            obs = self.current_obs if self.current_obs is not None else np.zeros(self.obs_dim, dtype=np.float32)
            terminated = False
            truncated = True
            reward = 0.0
            info = {"error": "No new observation in step()"}
            return obs, reward, terminated, truncated, info

        self.current_obs = obs
        self.step_count += 1

        # Compute reward (same as your compute_reward_from_state)
        reward = self._compute_reward_from_state(obs)

        # Termination logic: you want fixed-length episodes
        terminated = False
        truncated = (self.step_count >= self.max_steps) or self.node.is_truncated

        info = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        # Destroy ROS node on env close
        if self.node is not None:
            self.node.destroy_node()
        # rclpy.shutdown() should be called from the training script

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _wait_for_new_observation(self, timeout_sec: float) -> Optional[np.ndarray]:
        """Wait until a new ROS observation arrives (obs_seq changes)."""
        start_time = time.time()
        start_seq = self.node.obs_seq

        while time.time() - start_time < timeout_sec:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.node.obs_seq != start_seq and self.node.latest_obs is not None:
                self._last_obs_seq = self.node.obs_seq
                return self._obs_from_msg(self.node.latest_obs)

        return None

    def _obs_from_msg(self, msg: Observation) -> np.ndarray:
        """Convert Observation message to a 20-dim numpy array."""
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # 8 joint positions
        obs[0] = msg.left_leg_1_pos
        obs[1] = msg.right_leg_1_pos
        obs[2] = msg.left_leg_2_pos
        obs[3] = msg.right_leg_2_pos
        obs[4] = msg.left_leg_3_pos
        obs[5] = msg.right_leg_3_pos
        obs[6] = msg.left_leg_4_pos
        obs[7] = msg.right_leg_4_pos

        # 8 joint velocities
        obs[8]  = msg.left_leg_1_vel
        obs[9]  = msg.right_leg_1_vel
        obs[10] = msg.left_leg_2_vel
        obs[11] = msg.right_leg_2_vel
        obs[12] = msg.left_leg_3_vel
        obs[13] = msg.right_leg_3_vel
        obs[14] = msg.left_leg_4_vel
        obs[15] = msg.right_leg_4_vel

        # lidar: use index 1 as height
        if len(msg.lidar_ranges) > 1:
            obs[16] = msg.lidar_ranges[1]
        else:
            obs[16] = 0.0

        # imu_orientation: roll, pitch, (yaw)
        if len(msg.imu_orientation) >= 2:
            obs[17] = msg.imu_orientation[0]  # roll
            obs[18] = msg.imu_orientation[1]  # pitch
        else:
            obs[17] = 0.0
            obs[18] = 0.0

        # acceleration z (az)
        if len(msg.acceleration) >= 3:
            obs[19] = msg.acceleration[2]
        else:
            obs[19] = 0.0

        obs = np.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)

        return obs

    def _compute_reward_from_state(self, state_np: np.ndarray) -> float:
        """Same reward you used in ActorCriticNode."""
        height = float(state_np[16])
        roll   = float(state_np[17])
        pitch  = float(state_np[18])

        target_height = (self.height_limit_lower + self.height_limit_upper) / 2.0
        alive_bonus = 0.1

        reward = 0.0

        # 1) height reward
        if height < self.height_limit_lower or height > self.height_limit_upper:
            reward -= 5.0
        else:
            height_error = height - target_height
            reward += 3.0 * np.exp(- (height_error / 0.05) ** 2)

        # 2) orientation penalty
        angle_penalty_scale = 3.0
        reward -= angle_penalty_scale * (roll ** 2 + pitch ** 2)

        # 3) alive bonus
        reward += alive_bonus

        # 4) strong penalty on failure
        if self.node.is_truncated:
            reward -= 20.0

        return float(reward)