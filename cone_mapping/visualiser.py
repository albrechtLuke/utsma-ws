import cv2

def draw_detections(frame, dets, points):
    for ((x1, y1, x2, y2, _, cls), (X, Y, Z)) in zip(dets, points):
        u, v = (x1 + x2)//2, y2
        colour = (0, 0, 255) if cls == 0 else (0, 255, 0)
        cv2.circle(frame, (u, v), 5, colour, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    cv2.imshow('Cones', frame)
    return cv2.waitKey(1) != 27 # Exit on ESC Key press
        