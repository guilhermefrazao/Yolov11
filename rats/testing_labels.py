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
    #TODO fazer com que o código pegue automaticamente os valores no txt
    bounding_box = [[0.0, 0.5375000238418579, 0.08125001192092896, 0.2250000238418579, 0.1625000238418579]]
    image_path = 'datasets/augmentation/augmented_images/1_jpg.rf.6e8297a2a0b0174280aadf17cef2f1e5_augment.jpg'
    output_folder = 'datasets/augmentation/labeled_images'
    base_filename = os.path.splitext(os.path.basename(image_path))

    image = Image.open(image_path)
    image_rectangle = cv2.imread(image_path)

    draw_yolo(image_rectangle, bounding_box)

    output_path = os.path.join(output_folder, f"{base_filename}_labeled.jpg")

    cv2.imwrite(output_path, image_rectangle)

    cv2.imshow("Labeled_image", image_rectangle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


