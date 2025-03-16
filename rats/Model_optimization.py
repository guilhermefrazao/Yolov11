from ultralytics import YOLO

model = YOLO("runs/detect/train15_Yolov8_augmentation/weights/best.pt")

model.tune(data="datasets/data.yaml",epochs=200, optimizer="AdamW", device="cuda")