import rclpy
import numpy as np
import csv
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from base_class.base_node_class import DiabloBaseNode
from collections import defaultdict


class DiabloQLearningNode(DiabloBaseNode):
    def __init__(self):
        super().__init__("Q_learning")

        self.action_space = [
            np.zeros(8),                       # action 0: no torque
            np.ones(8) * 0.2,                  # action 1: small forward torque
            np.ones(8) * -0.2,                 # action 2: small backward torque
        ]
        self.n_actions = len(self.action_space)

        self.q_values = defaultdict(lambda: np.zeros(self.n_actions))

        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.discount_factor = 0.99
        self.start_epsilon = 1.0
        self.epsilon_decay = 0.99
        self.final_epsilon = 0.01    

    def digitize_state(self, observation):
        observation_array = np.array(observation)
        observation_array[0] = np.linspace(-0.1, 0.1, 10)
        observation_array[1] = np.linspace(-0.1, 0.1, 10)
        observation_array[2] = np.linspace(-0.7854, 0.7854,10)
        observation_array[3] = np.linspace(-0.7854, 0.7854,10)
        observation_array[4] = np.linspace(-0.7854, 0.7854,10)
        observation_array[5] = np.linspace(-0.7854, 0.7854,10)
        observation_array[6] = np.linspace(-3.14, 3.14, 10)
        observation_array[7] = np.linspace(-3.14, 3.14, 10)

        observation_array[8] = np.linspace(-3.14, 3.14, 10)
        observation_array[9] = np.linspace(-3.14, 3.14, 10)
        observation_array[10] = np.linspace(-3.14, 3.14, 10)
        observation_array[11] = np.linspace(-3.14,  3.14, 10)
        observation_array[12] = np.linspace(-3.14, 3.14, 10)
        observation_array[13] = np.linspace(-3.14, 3.14, 10)
        observation_array[14] = np.linspace(-3.14, 3.14, 10)
        observation_array[15] = np.linspace(-3.14, 3.14, 10)

        observation_array[17] = np.linspace(0.25, 0.35, 10)

        # Create a new array containing only the desired indices
        selected_obs_array = np.array([observation_array[i] for i in list(range(16)) + [17]])

        # Discretize using the original one-liner
        return tuple(np.digitize(observation[i], selected_obs_array[i]) for i in range(len(selected_obs_array)))
    

    def get_action(self, observation):
        state = self.digitize_state(observation)
        # epsilon = max(self.final_epsilon, self.start_epsilon * (self.epsilon_decay ** self.episode))
        if np.random.random() < self.start_epsilon:
            return np.random.randint(self.n_actions)  # Explore
        else:
            return np.argmax(self.q_values[state])  # Exploit



def main(args=None):
    rclpy.init(args=args)
    node = DiabloQLearningNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()