import ray
import time
import torchvision
import os
import torch 


from ultralytics import ASSETS, YOLO


@ray.remote(num_gpus=1)
def Yolov8_train():
    model = YOLO("yolov8n.pt", task="detect")
    
    results = model.train(data="C:/Users/guilh/Documents/GitHub_clone/Yolov11/datasets/data.yaml", device=0, epochs=100, imgsz=640, batch=8, workers=8, project="runs/detect/train15_Yolov8_augmentation", name="train15_Yolov8_augmentation", exist_ok=True)
    
    return results

def Yolov8_val():
    model = YOLO("runs/detect/train15_Yolov8_augmentation/weights/best_tuned.pt", task="detect")
    
    results = model.val(data="C:/Users/guilh/Documents/GitHub_clone/Yolov11/datasets/data.yaml", device=0, epochs=100, imgsz=640, batch=8, workers=8, project="runs/detect/train15_Yolov8_augmentation", name="train15_Yolov8_augmentation", exist_ok=True)
    
    return results


def Yolov8_test():
    model = YOLO("runs/detect/train15_Yolov8_augmentation/weights/best_tuned.pt", task="detect")
    
    results = model.test(data="C:/Users/guilh/Documents/GitHub_clone/Yolov11/datasets/data.yaml", device=0, epochs=100, imgsz=640, batch=8, workers=8, project="runs/detect/train15_Yolov8_augmentation", name="train15_Yolov8_augmentation", exist_ok=True)
    
    return results

def Run_remote():
    ray.init()

    ray.get([Yolov8_train.remote() for _ in range(os.cpu_count())])

if __name__ == "__main__":
    print(torch.__version__)
    print(torchvision.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)

    Run_remote()
