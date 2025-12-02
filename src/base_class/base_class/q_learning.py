import rclpy
import numpy as np
import csv
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from base_class.base_node_class import DiabloBaseNode


class DiabloQLearningNode(DiabloBaseNode):
    def __init__(self):
        super().__init__("q_learning")

        # -----------------------------
        # Q-LEARNING PARAMETERS
        # -----------------------------
        self.n_bins = 20                       # discretization bins for lidar[1]
        self.action_space = [
            np.zeros(8),                       # action 0: no torque
            np.ones(8) * 0.2,                  # action 1: small forward torque
            np.ones(8) * -0.2,                 # action 2: small backward torque
        ]
        self.n_actions = len(self.action_space)

        self.min_dist = 0.0                    # lidar min
        self.max_dist = 1.5                    # lidar max (safe)
        self.termination_height = 0.35         # episode ends if lidar[1] < 0.35

        # Q-table (distance_bin × actions)
        self.Q = np.zeros((self.n_bins, self.n_actions))

        # Learning params
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        # Episode bookkeeping
        self.episode = 0
        self.max_episodes = 3000
        self.total_reward = 0
        self.last_state = None
        self.last_action = None

        # CSV Logging
        self.csv_file = open("q_learning_log.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["episode", "step", "state", "action", "reward", "next_state"])

        # Timer to run RL loop
        self.timer = self.create_timer(0.05, self.training_step)

        self.get_logger().info("Q-learning node started.")

    # -----------------------------
    # DISCRETIZE STATE
    # -----------------------------
    def get_state(self):
        ground_dist = self.diablo_observations[17]  # lidar[1]
        clipped = np.clip(ground_dist, self.min_dist, self.max_dist)
        bin_idx = int((clipped - self.min_dist) / (self.max_dist - self.min_dist) * (self.n_bins - 1))
        return bin_idx

    # -----------------------------
    # EPSILON-GREEDY ACTION SELECTION
    # -----------------------------
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------
    def training_step(self):
        if self.episode >= self.max_episodes:
            self.get_logger().info("Training completed.")
            self.csv_file.close()
            return

        if not self.is_simulation_ready():
            return

        # Get discrete state from lidar[1]
        state = self.get_state()

        # Select action
        action_idx = self.choose_action(state)
        action_array = Float64MultiArray()
        action_array.data = self.action_space[action_idx].tolist()

        # Apply action
        self.take_action(action_array)

        # Reward
        ground = self.diablo_observations[17]
        reward = 1.0  # reward for staying alive

        done = False
        if ground < self.termination_height:
            reward = -50.0
            done = True

        self.total_reward += reward

        # Q-Learning update
        if self.last_state is not None:
            best_next = np.max(self.Q[state])
            td_target = reward + self.gamma * best_next * (0 if done else 1)
            td_error = td_target - self.Q[self.last_state][self.last_action]

            self.Q[self.last_state][self.last_action] += self.alpha * td_error

        # Log to CSV
        self.csv_writer.writerow([
            self.episode,
            self.last_state if self.last_state is not None else -1,
            state,
            action_idx,
            reward,
            state
        ])

        # Store for next step
        self.last_state = state
        self.last_action = action_idx

        if self.episode >= self.max_episodes:
            self.get_logger().info("Training completed.")

            np.save("q_table.npy", self.Q)
            self.get_logger().info("Saved Q-table to q_table.npy")

            self.csv_file.close()
            return

        # Episode finished
        if done:
            self.get_logger().info(f"Episode {self.episode} finished → Reward: {self.total_reward}")

            # Reset episode
            self.episode += 1
            self.total_reward = 0
            self.last_state = None
            self.last_action = None

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Reset simulation
            self.restart_learning_loop()




def main(args=None):
    rclpy.init(args=args)
    node = DiabloQLearningNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
