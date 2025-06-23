import cv2
from zed_interface import init_camera, grab_data
from onnx_inference import ONNXDetector
from projector import project_to_3d
from publisher import init_node
from visualiser import draw_detections

def main():
    zed, runtime = init_camera()
    detector = ONNXDetector('models/bestv8.onnx')
    node = init_node()

    try:
        while True:
            image, point_cloud = grab_data(zed, runtime)
            if image is None:
                continue
            
            frame = image.get_data()[:, :, :3]
            dets = detector.infer(frame)


            if dets is not None or len(dets) > 0:
                print(f"Detected {len(dets)} cones.")
                print(dets)
                points = []
                for det in dets:
                    if len(det) < 6:  
                        continue

                    x1, y1, x2, y2 = map(int, det[:4])
                    conf = det[4]
                    cls = det[5]

                    u, v = (x1+x2)//2, y2
                    X, Y, Z = project_to_3d(point_cloud, u, v)
                    points.append((X, Y, Z))
                    node.publish_cone_pose(X, Y, Z)
            else:
                points = []

            if not draw_detections(frame, dets, points):
                break

    finally:
        if zed:
            zed.close()
        cv2.destroyAllWindows()
        print("All resources released.")

if __name__ == '__main__':
    main()
