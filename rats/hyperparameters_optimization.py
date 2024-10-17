from ultralytics import YOLO

model = YOLO("yolov8s-world.pt")

model.tune(data="datasets/data.yaml",epochs=200, optimizer="AdamW", device="cuda")