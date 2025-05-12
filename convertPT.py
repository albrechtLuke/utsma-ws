from ultralytics import YOLO

model = YOLO("bestv12.pt")

model.export(format="onnx")