from ultralytics import YOLO

model = YOLO("runs/detect/train15_Yolov8_augmentation/weights/best.pt")

space = {
    "lr0": (1e-3, 1e-4, 1e-5),
    "lrf": (0.01, 0.1, 0.2),
    "degrees": (0, 90, 180),
    "momentum": (0.9, 0.95, 0.99),
    "weight_decay": (0.0005, 0.001, 0.01),
    "epochs": (50, 100, 200),


}

model.tune(data="datasets/data.yaml",epochs=200, optimizer="AdamW",space=space, device="cuda")
#best_hyperparameters.yaml