import cv2

img_height = 480
img_width = 640
path = 'datasets/train/labels_augmentation/augmented_image_0_0.txt'
path_img = 'datasets/train/images_augmentation/augmented_image_0_3.png'

with open(path,'r') as file:
    line = file.readlines()

label = line[0].split()

class_id = label[0]
x_center = float(label[1]) * img_width
y_center = float(label[2]) * img_height
box_width = float(label[3]) * img_width
box_height = float(label[4]) * img_height

print(x_center, y_center, box_width, box_height)

x_min = int(x_center - (box_width / 2))
x_max = int(x_center + (box_width / 2))
y_min = int(y_center - (box_height / 2))
y_max = int(y_center + (box_height / 2))

print(x_min, y_min, x_max, y_max)


img = cv2.imread(path_img)

cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255,0,0), 2)

cv2.imshow("Labeled_image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()