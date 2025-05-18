import fiftyone.zoo as foz


dataset = foz.load_zoo_dataset("open-images-v7", label_types=["detections"], classes=["Mouse"])