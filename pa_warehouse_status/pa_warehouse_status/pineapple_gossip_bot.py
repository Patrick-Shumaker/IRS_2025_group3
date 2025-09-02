import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PineappleGossipBot(Node):
    def __init__(self):
        super().__init__('pineapple_gossip_bot') # Node name

        # Create publisher: topic_name, message_type, queue_size
        self.publisher_ = self.create_publisher(String, 'status_updates',10)

        # Timer to publish every 0.5 seconds
        timer_period = 2
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'yo {self.i}!'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args) # Initialize ROS2
    pineapple_gossip_bot = PineappleGossipBot()
    rclpy.spin(pineapple_gossip_bot) # Keep node running
    pineapple_gossip_bot.destroy_node()
    rclpy.shutdown()