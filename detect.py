import cv2
import numpy as np
import time as t
import pyzed.sl as sl 

def zed_init():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"ZED Error: {status}")
        exit(1)

    runtime_params = sl.RuntimeParameters()
    image_zed = sl.Mat()

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)

            frame = image_zed.get_data()

            # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            cv2.imshow("ZED | Press ESC to quit", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                print("Exiting...")
                break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    zed_init()
