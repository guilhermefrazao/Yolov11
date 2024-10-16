from ultralytics import YOLO

model = YOLO("yolov8s-world.pt")

model.tune(data="datasets/data.yaml",epochs=200, interations=300, optimizer="AdamW", device="cuda", batch_size=16)