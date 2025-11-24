from base_class.base_node_class import DiabloBaseNode
import rclpy
from std_msgs.msg import Float64
import numpy as np
import control

class LinearQuadraticControlNode(DiabloBaseNode):
    def __init__ (self):
        super().__init__(node_name="controller")
        
        state_matrix = self.get_state_matrix()
        input_matrix = self.get_input_matrix()

        Q = self.get_state_weighing_matrix()  # State cost matrix
        R = self.get_control_weighing_matrix()   # Input cost matrix
        self.controller, self.solution_matrix, self.eigen_values = control.lqr(state_matrix, input_matrix, Q, R)

    def get_state_matrix(self) -> np.ndarray:
        A = np.zeros((16, 16))
        # Position derivatives
        A[0, 8] = 1.0
        A[1, 9] = 1.0
        A[2, 10] = 1.0
        A[3, 11] = 1.0
        A[4, 12] = 1.0
        A[5, 13] = 1.0
        A[6, 14] = 1.0
        A[7, 15] = 1.0
        # Velocity derivatives (assuming simple dynamics for illustration)
        # In a real scenario, these would be derived from the robot's dynamics
        return A
    
    def get_input_matrix(self) -> np.ndarray:
        B = np.zeros((16, 8))
        # Assuming each control input directly affects the corresponding joint velocity
        B[8, 0] = 1.0
        B[9, 1] = 1.0
        B[10, 2] = 1.0
        B[11, 3] = 1.0
        B[12, 4] = 1.0
        B[13, 5] = 1.0
        B[14, 6] = 1.0
        B[15, 7] = 1.0
        return B
    

    def get_state_weighing_matrix(self) -> np.ndarray:
        return np.diag([10.0, 
                     10.0, 
                     10.0, 
                     10.0, 
                     1.0, 
                     1.0, 
                     1.0, 
                     1.0,
                     0.1, 
                     0.1, 
                     0.1, 
                     0.1, 
                     0.01, 
                     0.01, 
                     0.01, 
                     0.01
        ])
    
    def get_control_weighing_matrix(self) -> np.ndarray:
        return np.diag([
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01
        ])
    
    def get_control_value(self) -> Float64:
        return -np.dot(self.controller, np.array(self.diablo_observations))
    
    def run_simulation(self):
        if not self.is_simulation_ready():
            return

        if self.is_simulation_stopped():
            self.restart_learning_loop()
            self.get_logger().info("Simulation stopped. Restarting...")
            return

        self.take_action(Float64(data=self.get_control_value()))

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(LinearQuadraticControlNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()

