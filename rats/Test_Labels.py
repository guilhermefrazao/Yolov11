import cv2
import pybboxes as pbx
import os

from PIL import Image

def draw_yolo(image, labels):
    """ function to draw bounding box in the augmented image"""
    H, W = image.shape[:2]

    for label in labels:
        yolo_normalized = label[1:]
        box_voc = pbx.convert_bbox(tuple(yolo_normalized), from_type="yolo", to_type="voc", image_size=(W, H))
        cv2.rectangle(image, (box_voc[0], box_voc[1]),(box_voc[2], box_voc[3]), (0, 255, 0), 2)
    

if __name__ == "__main__":
    with open('datasets/roboflow/train/labels/16_png.rf.d2ad615007ee3f099964b7797252e941.txt', 'r') as file:
        file_data = file.readlines()
        bounding_box = [list(map(float, line.strip().split())) for line in file_data]

    image_path = 'datasets/roboflow/train/images/16_png.rf.d2ad615007ee3f099964b7797252e941.jpg'

    image_rectangle = cv2.imread(image_path)

    draw_yolo(image_rectangle, bounding_box)

    cv2.imshow("Labeled_image", image_rectangle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()