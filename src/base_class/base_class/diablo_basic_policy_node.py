import rclpy
from base_class.reinforcement_learning_node import ReinforcementLearningNode


class DiabloReinforcementBasicPolicy(ReinforcementLearningNode):
    def __init__(self):
        super().__init__("diablo_basic_policy_node")
        self.create_timer(0.05, self.run)

    def run_one_step(self):
        
        current_obs = self.get_diablo_observations()
        ground_distance = current_obs[17]
        
        self.get_logger().info(f"Step {self.step}: Ground distance = {ground_distance}, Truncated = {self.is_simulation_stopped()}")


        if self.is_simulation_stopped():
            self.get_logger().info("Simulation truncated. Restarting...")
            self.restart_learning_loop()
            return
        
        self.take_action(self.create_command(int(self.get_diablo_observations()[17] < 0.35)))
        self.step += 1

    def run(self):
        if not self.is_simulation_ready():
            return

        self.stop_run_when_learning_ended()
        self.advance_episode_when_finished()
        self.run_one_step()


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(DiabloReinforcementBasicPolicy())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
