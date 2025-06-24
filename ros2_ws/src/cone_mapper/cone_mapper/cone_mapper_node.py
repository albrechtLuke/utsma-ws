import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Point, Pose, PoseArray, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from zed_msgs.msg import ObjectsStamped
import tf2_ros
import tf2_geometry_msgs
from rclpy.duration import Duration
from scipy.spatial import Delaunay

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
        self.declare_parameter('max_cone_age', 2.0)
        self.declare_parameter('association_threshold', 3.0)
        self.declare_parameter('camera_height', 0.5)

        self.fixed_frame = self.get_parameter('fixed_frame').value
        self.max_cone_age = self.get_parameter('max_cone_age').value
        self.association_threshold = self.get_parameter('association_threshold').value
        self.camera_height = self.get_parameter('camera_height').value

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        

        self.cones = {}
        self.cone_counter = 0
        self.cone_dimensions = {
            'blue_cone': (0.3, 0.3),
            'yellow_cone': (0.3, 0.3)
        }

        self.triangulation_pub = self.create_publisher(MarkerArray, '/cone_map/triangulation', 10)

        self.create_subscription(ObjectsStamped, '/zed/zed_node/obj_det/objects', self.detection_callback, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/cone_map/markers', 10)
        self.poses_pub = self.create_publisher(PoseArray, '/cone_map/poses', 10)
        self.start_time = self.get_clock().now()
        self.create_timer(0.1, self.publish_map)

    def detection_callback(self, msg):
        if (self.get_clock().now() - self.start_time).nanoseconds / 1e9 < 1.0:
            return
        current_time = self.get_clock().now()
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame=self.fixed_frame,
                source_frame=msg.header.frame_id,
                time=rclpy.time.Time(),
                timeout=Duration(seconds=1)
            )
        except Exception:
            return

        for obj in msg.objects:
            if obj.label not in ['blue_cone', 'yellow_cone']:
                continue
            if hasattr(obj, 'confidence') and obj.confidence < 0.8:
                continue
            if np.linalg.norm([obj.position[0], obj.position[1]]) > 15.0 or obj.position[2] > self.camera_height + 1.0:
                continue

            base_position = self.calculate_cone_base(obj)
            transformed_position = self.transform_point(base_position, msg.header.frame_id, msg.header.stamp, transform)
            if transformed_position is None:
                continue

            cone_id = self.associate_detection(obj.label, transformed_position)
            if cone_id is not None:
                self.update_cone(cone_id, transformed_position, current_time)
            elif self.is_valid_cone_position(obj.label, transformed_position):
                self.create_new_cone(obj.label, transformed_position, current_time)

    def is_valid_cone_position(self, label, position):
        points = []
        labels = []
        # append pos and type of confirmed cones
        for cone in self.cones.values():
            if not cone['confirmed']:
                continue
            pos = cone['kf'].get_position()
            points.append(pos)
            labels.append(cone['label'])

        points.append(position)
        labels.append(label)

        # check for valid triangulation
        if len(points) < 3:
            return True
        try:
            tri = Delaunay(points)
        except Exception:
            return True

        # Check if the triangulation follows rules of blue ouside line, yellow inside line
        # discard triangles that do not follow this rule
        for simplex in tri.simplices:
            triangle_labels = [labels[i] for i in simplex]
            if 'blue_cone' in triangle_labels and 'yellow_cone' in triangle_labels:
                if len(set(triangle_labels)) == 2:
                    return True  # this triangle shows a good left-right pairing
                else:
                    print(f"Invalid triangle detected with labels: {triangle_labels}")
                    return False  # invalid triangle with mixed labels
        return False

    def filter_valid_triangles(self, points, labels, simplices):
        valid_simplices = []

        for simplex in simplices:
            triangle_labels = [labels[i] for i in simplex]
            triangle_points = [points[i] for i in simplex]

            # Require at least one blue and one yellow
            if 'blue_cone' in triangle_labels and 'yellow_cone' in triangle_labels:
                # Get yellow and blue cone indices
                yellow_idx = [i for i in simplex if labels[i] == 'yellow_cone'][0]
                blue_idx = [i for i in simplex if labels[i] == 'blue_cone'][0]

                yellow_pos = np.array(points[yellow_idx])
                blue_pos = np.array(points[blue_idx])
                mid = np.mean([points[i] for i in simplex], axis=0)

                # Vector from yellow to blue should point outward
                direction = blue_pos - yellow_pos
                center_dir = mid - yellow_pos

                angle = np.arccos(np.dot(direction, center_dir) / (np.linalg.norm(direction) * np.linalg.norm(center_dir) + 1e-6))

                if angle < np.pi / 2:  # less than 90 degrees
                    valid_simplices.append(simplex)

        return valid_simplices


    def calculate_cone_base(self, obj):
        try:
            height = obj.dimensions_3d[2] if hasattr(obj, 'dimensions_3d') and len(obj.dimensions_3d) > 2 else 0.3
            return [obj.position[0], obj.position[1], obj.position[2] - height / 2.0]
        except Exception:
            return [obj.position[0], obj.position[1], obj.position[2]]

    def transform_point(self, point, source_frame, stamp, transform):
        try:
            pt = PointStamped()
            pt.header.frame_id = source_frame
            pt.header.stamp = stamp
            pt.point = Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
            transformed = tf2_geometry_msgs.do_transform_point(pt, transform)
            return [transformed.point.x, transformed.point.y]
        except Exception:
            return None

    def associate_detection(self, label, position):
        best_match = None
        min_distance = float('inf')
        for cone_id, cone in self.cones.items():
            if cone['label'] != label:
                continue
            dist = cone['kf'].mahalanobis_distance(position)
            if dist < self.association_threshold and dist < min_distance:
                min_distance = dist
                best_match = cone_id
        return best_match

    def update_cone(self, cone_id, position, update_time):
        cone = self.cones[cone_id]
        prev_pos = cone['kf'].get_position()
        if np.linalg.norm(np.array(prev_pos) - np.array(position)) > 2.0:
            return
        cone['kf'].predict()
        cone['kf'].update(position)
        cone['last_update'] = update_time
        cone['detections'] += 1
        if cone['detections'] >= 3 and (update_time - cone['first_seen']).nanoseconds / 1e9 >= 1.0:
            cone['confirmed'] = True

    def create_new_cone(self, label, position, update_time):
        self.cone_counter += 1
        self.cones[self.cone_counter] = {
            'label': label,
            'kf': ConeKalmanFilter(position),
            'last_update': update_time,
            'first_seen': update_time,
            'detections': 1,
            'confirmed': False
        }
    
    def publish_triangulation(self):
        marker_array = MarkerArray()
        base_time = self.get_clock().now().to_msg()

        # Line marker
        line_marker = Marker()
        line_marker.header.frame_id = self.fixed_frame
        line_marker.header.stamp = base_time
        line_marker.ns = "triangles"
        line_marker.id = 0
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.02
        line_marker.color.r = 0.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0

        # Get confirmed cone positions and IDs
        confirmed_positions = []
        confirmed_ids = []
        labels = []
        for cone_id, cone in self.cones.items():
            if cone['confirmed']:
                confirmed_positions.append(cone['kf'].get_position())
                confirmed_ids.append(cone_id)
                labels.append(cone['label'])

        if len(confirmed_positions) < 3:
            return

        try:
            tri = Delaunay(confirmed_positions)
        except Exception:
            return

        text_id = 1000  # starting ID for text markers

        points = confirmed_positions
        valid_simplices = self.filter_valid_triangles(points, labels, tri.simplices)
        for simplex in valid_simplices:
            indices = [confirmed_ids[i] for i in simplex]
            positions = [confirmed_positions[i] for i in simplex]
            for i in range(3):
                a = np.array(positions[i])
                b = np.array(positions[(i + 1) % 3])
                midpoint = (a + b) / 2.0
                length = np.linalg.norm(a - b)

                # Add the line to the LINE_LIST marker
                line_marker.points.append(Point(x=a[0], y=a[1], z=0.0))
                line_marker.points.append(Point(x=b[0], y=b[1], z=0.0))

                # Add a TEXT marker for this line
                text_marker = Marker()
                text_marker.header.frame_id = self.fixed_frame
                text_marker.header.stamp = base_time
                text_marker.ns = "triangle_text"
                text_marker.id = text_id
                text_id += 1
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.scale.z = 0.1  # text height
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.color.a = 0.8
                text_marker.pose.position.x = midpoint[0]
                text_marker.pose.position.y = midpoint[1]
                text_marker.pose.position.z = 0.1  # above the line
                text_marker.pose.orientation.w = 1.0

                text_marker.text = f"{length:.2f}m"
                marker_array.markers.append(text_marker)

        marker_array.markers.append(line_marker)
        self.triangulation_pub.publish(marker_array)




    def unconfirm_long_edge_cones(self, max_edge_length):
        confirmed_ids = []
        points = []
        id_list = []

        for cone_id, cone in self.cones.items():
            if cone['confirmed']:
                points.append(cone['kf'].get_position())
                id_list.append(cone_id)

        if len(points) < 3:
            return

        try:
            tri = Delaunay(points)
        except Exception:
            return

        # cone edge lengths each cone participates in
        cone_edge_lengths = {cone_id: [] for cone_id in id_list}

        for simplex in tri.simplices:
            indices = [id_list[i] for i in simplex]
            positions = [points[i] for i in simplex]
            for i in range(3):
                a, b = positions[i], positions[(i+1)%3]
                id_a, id_b = indices[i], indices[(i+1)%3]
                edge_length = np.linalg.norm(np.array(a) - np.array(b))
                cone_edge_lengths[id_a].append(edge_length)
                cone_edge_lengths[id_b].append(edge_length)

        # Unconfirm cones where all edge lengths are too long
        for cone_id, lengths in cone_edge_lengths.items():
            if all(length > max_edge_length for length in lengths):
                self.cones[cone_id]['confirmed'] = False
                print(f"Unconfirming cone {cone_id} due to long edges: {lengths}")



    def publish_map(self):
        current_time = self.get_clock().now()
        self.remove_old_cones(current_time)

        marker_array = MarkerArray()
        pose_array = PoseArray()
        pose_array.header.stamp = current_time.to_msg()
        pose_array.header.frame_id = self.fixed_frame

        for cone_id, cone in self.cones.items():
            if cone['detections'] < 3:
                continue

            pos = cone['kf'].get_position()
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
            marker.id = cone_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.lifetime = Duration(seconds=0).to_msg() if cone['confirmed'] else Duration(seconds=0.2).to_msg()

            diameter, height = self.cone_dimensions[cone['label']]
            marker.scale.x = diameter
            marker.scale.y = diameter
            marker.scale.z = height

            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = height / 2
            marker.pose.orientation.w = 1.0

            if cone['label'] == 'blue_cone':
                marker.color.r = 0.1
                marker.color.g = 0.1
                marker.color.b = 1.0
            else:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            marker.color.a = 0.7

            marker_array.markers.append(marker)

        # self.unconfirm_long_edge_cones(max_edge_length=1.5)


        self.marker_pub.publish(marker_array)
        self.poses_pub.publish(pose_array)
        self.publish_triangulation()


    def remove_old_cones(self, current_time):
        to_remove = [
            cone_id for cone_id, cone in self.cones.items()
            if not cone['confirmed'] and
               (current_time - cone['last_update']).nanoseconds / 1e9 > self.max_cone_age
        ]
        for cone_id in to_remove:
            del self.cones[cone_id]

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