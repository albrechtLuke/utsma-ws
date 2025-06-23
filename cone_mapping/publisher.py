import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped

class ConePublisher(Node):
    def __init__(self):
        super().__init__('cone_publisher')
        self.pub = self.create_publisher(PointStamped, '/cone_points', 10)

    def publish_cone_pose(self, x, y, z, frame_id='zed_left_camera_frame'):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.point.x, msg.point.y, msg.point.z = x, y, z
        self.pub.publish(msg)

def init_node():
    rclpy.init()
    return ConePublisher()