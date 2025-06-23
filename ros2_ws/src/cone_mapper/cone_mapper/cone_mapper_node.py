import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Point, Pose, PoseArray, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from zed_msgs.msg import ObjectsStamped
import tf2_ros
import tf2_geometry_msgs
import math
from rclpy.duration import Duration

class ConeKalmanFilter:
    def __init__(self, initial_pos):
        self.state = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=np.float32).reshape(4, 1)
        self.covariance = np.eye(4) * 0.1
        dt = 0.1
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 0.1

    def predict(self):
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q

    def update(self, measurement):
        z = np.array(measurement).reshape(2, 1)
        y = z - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.covariance = (np.eye(4) - K @ self.H) @ self.covariance

    def get_position(self):
        return self.state[:2].flatten().tolist()

    def mahalanobis_distance(self, measurement):
        z = np.array(measurement).reshape(2, 1)
        y = z - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + self.R
        return float(y.T @ np.linalg.inv(S) @ y)

class ConeMapper(Node):
    def __init__(self):
        super().__init__('cone_mapping_node')
        self.declare_parameter('fixed_frame', 'map')
        self.declare_parameter('max_track_age', 2.0)
        self.declare_parameter('association_threshold', 3.0)
        self.declare_parameter('camera_height', 0.5)

        self.fixed_frame = self.get_parameter('fixed_frame').value
        self.max_track_age = self.get_parameter('max_track_age').value
        self.association_threshold = self.get_parameter('association_threshold').value
        self.camera_height = self.get_parameter('camera_height').value

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.tracks = {}
        self.track_counter = 0
        self.cone_dimensions = {
            'blue_cone': (0.3, 0.3),
            'yellow_cone': (0.3, 0.3)
        }

        self.create_subscription(ObjectsStamped, '/zed/zed_node/obj_det/objects', self.detection_callback, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/cone_map/markers', 10)
        self.poses_pub = self.create_publisher(PoseArray, '/cone_map/poses', 10)
        self.start_time = self.get_clock().now()
        self.create_timer(0.1, self.publish_map)

    def detection_callback(self, msg):
        if (self.get_clock().now() - self.start_time).nanoseconds / 1e9 < 1.0:
            self.get_logger().info("Waiting for TF to initialize...")
            return
        current_time = self.get_clock().now()
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame=self.fixed_frame,
                source_frame=msg.header.frame_id,
                time=rclpy.time.Time(),
                timeout=Duration(seconds=1)
            )
        except Exception as e:
            self.get_logger().warn(f'Transform unavailable: {str(e)}')
            return

        for obj in msg.objects:
            if obj.label not in ['blue_cone', 'yellow_cone']:
                continue
            if hasattr(obj, 'confidence') and obj.confidence < 0.6:
                continue
            if np.linalg.norm([obj.position[0], obj.position[1]]) > 15.0 or obj.position[2] > self.camera_height + 1.0:
                continue

            base_position = self.calculate_cone_base(obj)
            transformed_position = self.transform_point(base_position, msg.header.frame_id, msg.header.stamp, transform)
            if transformed_position is None:
                continue

            track_id = self.associate_detection(obj.label, transformed_position)
            if track_id is not None:
                self.update_track(track_id, transformed_position, current_time)
            else:
                self.create_new_track(obj.label, transformed_position, current_time)

    def calculate_cone_base(self, obj):
        try:
            height = obj.dimensions_3d[2] if hasattr(obj, 'dimensions_3d') and len(obj.dimensions_3d) > 2 else (0.3 if obj.label == 'blue_cone' else 0.2)
            return [obj.position[0], obj.position[1], obj.position[2] - height/2.0]
        except Exception as e:
            self.get_logger().error(f'Cone base calculation failed: {str(e)}')
            return [obj.position[0], obj.position[1], obj.position[2]]

    def transform_point(self, point, source_frame, stamp, transform):
        try:
            pt = PointStamped()
            pt.header.frame_id = source_frame
            pt.header.stamp = stamp
            pt.point = Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
            transformed = tf2_geometry_msgs.do_transform_point(pt, transform)
            return [transformed.point.x, transformed.point.y]
        except Exception as e:
            self.get_logger().error(f'Point transform failed: {str(e)}')
            return None

    def associate_detection(self, label, position):
        best_match = None
        min_distance = float('inf')
        for track_id, track in self.tracks.items():
            if track['label'] != label:
                continue
            dist = track['kf'].mahalanobis_distance(position)
            if dist < self.association_threshold and dist < min_distance:
                min_distance = dist
                best_match = track_id
        return best_match

    def update_track(self, track_id, position, update_time):
        track = self.tracks[track_id]
        prev_pos = track['kf'].get_position()
        if np.linalg.norm(np.array(prev_pos) - np.array(position)) > 1.0:
            return
        track['kf'].predict()
        track['kf'].update(position)
        track['last_update'] = update_time
        track['detections'] += 1
        if track['detections'] >= 3:
            track['confirmed'] = True

    def create_new_track(self, label, position, update_time):
        self.track_counter += 1
        self.tracks[self.track_counter] = {
            'label': label,
            'kf': ConeKalmanFilter(position),
            'last_update': update_time,
            'detections': 1,
            'confirmed': False
        }

    def publish_map(self):
        current_time = self.get_clock().now()
        self.remove_old_tracks(current_time)

        marker_array = MarkerArray()
        pose_array = PoseArray()
        pose_array.header.stamp = current_time.to_msg()
        pose_array.header.frame_id = self.fixed_frame

        for track_id, track in self.tracks.items():
            if track['detections'] < 3:
                continue

            pos = track['kf'].get_position()
            pose = Pose()
            pose.position.x = pos[0]
            pose.position.y = pos[1]
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

            marker = Marker()
            marker.header.stamp = current_time.to_msg()
            marker.header.frame_id = self.fixed_frame
            marker.ns = "cones"
            marker.id = track_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.lifetime = Duration(seconds=0).to_msg() if track.get('confirmed', False) else Duration(seconds=0.2).to_msg()

            diameter, height = self.cone_dimensions[track['label']]
            marker.scale.x = diameter
            marker.scale.y = diameter
            marker.scale.z = height

            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = height/2
            marker.pose.orientation.w = 1.0

            if track['label'] == 'blue_cone':
                marker.color.r = 0.1
                marker.color.g = 0.1
                marker.color.b = 1.0
            else:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            marker.color.a = 0.7

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)
        self.poses_pub.publish(pose_array)

    def remove_old_tracks(self, current_time):
        to_remove = [
            track_id for track_id, track in self.tracks.items()
            if not track.get('confirmed', False) and
               (current_time - track['last_update']).nanoseconds / 1e9 > self.max_track_age
        ]
        for track_id in to_remove:
            del self.tracks[track_id]

def main(args=None):
    rclpy.init(args=args)
    node = ConeMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
