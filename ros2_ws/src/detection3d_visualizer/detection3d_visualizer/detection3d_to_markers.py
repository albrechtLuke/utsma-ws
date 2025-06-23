import rclpy
from rclpy.node import Node
from zed_msgs.msg import ObjectsStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from builtin_interfaces.msg import Duration
import random


class ZEDObjectsToMarkers(Node):
    def __init__(self):
        super().__init__('zed_objects_to_markers')

        self.publisher = self.create_publisher(MarkerArray, '/zed/bbox_markers', 10)
        self.subscription = self.create_subscription(
            ObjectsStamped,
            '/zed/zed_node/obj_det/objects',
            self.listener_callback,
            10
        )
        self.marker_lifetime = Duration(sec=0)  # 0 = forever

    def listener_callback(self, msg):
        markers = MarkerArray()

        for i, obj in enumerate(msg.objects):
            marker = Marker()
            marker.header = msg.header
            marker.ns = 'zed_objects'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Position and orientation
            marker.pose.position.x = float(obj.position[0])
            marker.pose.position.y = float(obj.position[1])
            marker.pose.position.z = float(obj.position[2])
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            # Dimensions
            dims = obj.dimensions_3d
            marker.scale.x = float(dims[0]) if len(dims) > 0 else 0.1
            marker.scale.y = float(dims[1]) if len(dims) > 1 else 0.1
            marker.scale.z = float(dims[2]) if len(dims) > 2 else 0.1

            # Color (yellow cone = yellow-ish)
            marker.color.r = 1.0 if obj.label == 'yellow_cone' else 0.0
            marker.color.g = 1.0 if obj.label == 'yellow_cone' else 0.0
            marker.color.b = 0.0 if obj.label == 'yellow_cone' else 1.0
            marker.color.a = 0.5

            marker.lifetime = self.marker_lifetime
            markers.markers.append(marker)

        self.publisher.publish(markers)
        self.get_logger().info(f'Published {len(markers.markers)} markers')


def main(args=None):
    rclpy.init(args=args)
    node = ZEDObjectsToMarkers()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

