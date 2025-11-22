import rclpy
from rclpy.node import Node
import torch


class TorchTestNode(Node):
    def __init__(self):
        super().__init__('venv_check')

        # Simple tensor operation
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        c = a + b

        self.get_logger().info(f"PyTorch works! a + b = {c}")


def main(args=None):
    rclpy.init(args=args)
    node = TorchTestNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
