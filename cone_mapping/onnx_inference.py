import onnxruntime as ort
import cv2
import numpy as np

class ONNXDetector:
    def __init__(self, model_path, providers=None):

        available_providers = ort.get_available_providers()
        print(f"Available ONNX Runtime providers: {available_providers}")
        providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available_providers else ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()

        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=providers
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        _, _, self.height, self.width = inp.shape

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.width, self.height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))[None, ...]
    
    def infer(self, frame):
        x = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: x})
        detections = outputs[0]

        if detections.ndim == 3:
            detections = detections[0]
        return detections