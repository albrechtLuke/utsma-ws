import argparse
import pyzed.sl as sl
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2


# ========================== YOLO TensorRT Inference Class ============================= #
class YOLOTRT:
    def __init__(self, engine_path, input_shape=(640, 640)):
        self.engine_path = engine_path
        self.input_shape = input_shape  # (W, H)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self._load_engine()
        self._allocate_buffers()

    def _load_engine(self):
        with open(self.engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_binding_idx = self.engine.get_binding_index(self.engine[0])
        self.output_binding_idx = self.engine.get_binding_index(self.engine[1])

    def _allocate_buffers(self):
        # Input
        self.input_shape_trt = trt.volume(self.engine.get_binding_shape(0))
        self.input_host = cuda.pagelocked_empty(self.input_shape_trt, dtype=np.float32)
        self.input_device = cuda.mem_alloc(self.input_host.nbytes)

        # Output
        output_shape = self.engine.get_binding_shape(1)
        self.output_shape = (output_shape[0], output_shape[1])
        self.output_host = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
        self.output_device = cuda.mem_alloc(self.output_host.nbytes)

        self.stream = cuda.Stream()

    def preprocess(self, image):
        """Resize and normalize image to model input."""
        resized = cv2.resize(image, self.input_shape)
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0  # Normalize to [0,1]
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # Add batch dim
        return np.ascontiguousarray(img)

    def infer(self, image_bgr):
        input_data = self.preprocess(image_bgr)
        np.copyto(self.input_host, input_data.ravel())

        # Transfer input to device, run inference, transfer output back
        cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)
        self.context.execute_async_v2(bindings=[
            int(self.input_device),
            int(self.output_device)
        ], stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
        self.stream.synchronize()

        output = self.output_host.reshape(self.output_shape)
        return self.postprocess(output)

    def postprocess(self, output, conf_thresh=0.5):
        """Filter out low-confidence detections and return bounding boxes."""
        results = []
        for det in output:
            x1, y1, x2, y2, conf, cls = det[:6]
            if conf < conf_thresh:
                continue
            results.append([int(x1), int(y1), int(x2), int(y2), int(cls), float(conf)])
        return results


# ============================ MAIN SCRIPT ENTRYPOINT ================================ #
def main():
    parser = argparse.ArgumentParser(description="YOLO TensorRT + ZED Perception Script")
    parser.add_argument('--engine', type=str, required=True, help='Path to TensorRT engine file')
    args = parser.parse_args()

    yolo = YOLOTRT(engine_path=args.engine, input_shape=(640, 640))

    # ==== INITIALIZE ZED CAMERA ====
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print("ZED initialization failed:", status)
        exit(1)

    image_zed = sl.Mat()
    depth_map = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    while True:
        start_total = time.time()

        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            img_rgb = image_zed.get_data()[:, :, :3]  # HWC, RGB
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

            # Inference
            start_inference = time.time()
            detections = yolo.infer(img_bgr)
            inference_time = (time.time() - start_inference) * 1000  # ms

            for det in detections:
                x1, y1, x2, y2, class_id, conf = det
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                err, point = depth_map.get_value(cx, cy)
                if err == sl.ERROR_CODE.SUCCESS:
                    x, y, z = point
                    label = f"ID:{class_id} ({conf:.2f}) {x:.1f},{y:.1f},{z:.1f}m"
                else:
                    label = f"ID:{class_id} ({conf:.2f})"

                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_bgr, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            total_time = (time.time() - start_total) * 1000
            cv2.putText(img_bgr, f"Inference: {inference_time:.1f} ms", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(img_bgr, f"Total: {total_time:.1f} ms", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("ZED Cone Detection", img_bgr)
            key = cv2.waitKey(1)
            if key == 27:
                break

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

