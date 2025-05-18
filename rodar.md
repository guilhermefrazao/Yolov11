Yolov_8 model comand run train: 
yolo task=detect  mode=train  model=ultralytics\cfg\models\v8\yolov8-p2.yaml  data=C:/Users/guilh/Documents/GitHub_clone/Yolov11/datasets/data.yaml  epochs=50 batch=8 device=0 optimizer=AdamW verbose=True

Yolov8 model comand run val:
yolo task=detect  mode=val  model=runs/detect/train1/weights/best.pt  data=C:/Users/guilh/Documents/GitHub_clone/Yolov11/datasets/data.yaml device=0 verbose=True

Yolov8 model comand run test:
yolo predict model=runs/detect/train1/weights/best.pt  source=datasets/roboflow/test/images imgsz=640   

Best metrics:
{'metrics/precision(B)': 0.92539, 'metrics/recall(B)': 0.88372, 'metrics/mAP50(B)': 0.93892, 'metrics/mAP50-95(B)': 0.62194, 'val/box_loss': 1.28078, 'val/cls_loss': 0.63854, 'val/dfl_loss': 1.67974, 'fitness': 0.65364}

Dowloading Images:
import fiftyone as fo
import fiftyone.zoo as foz


dataset = foz.load_zoo_dataset("open-images-v7", label_types=["detections"], classes=["Mouse"])

runs\detect\train5