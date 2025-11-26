import rclpy
from base_class.reinforcement_learning_node import ReinforcementLearningNode


class DiabloReinforcementBasicPolicy(ReinforcementLearningNode):
    def __init__(self):
        super().__init__("diablo_basic_policy_node")
        self.create_timer(0.05, self.run)

    def run_one_step(self):
        self.take_action(self.create_command(int(self.get_diablo_observations()[1] < 0)))
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
