from glob import glob
from functools import reduce
from xml.etree import ElementTree as et
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pybboxes as pbx
import albumentations as A
import cv2
import random
import numpy as np
import pandas as pd
import os

labels_dir = "datasets/train/labels"

txt_list = glob(labels_dir + '/**/*.txt', recursive = True)

    # Function to read and process the file
def extract_data(file_path):
    # Initialize an empty list to store the data
    data = []
    
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Read all lines in the file
        lines = file.readlines()
        
        # Process each line
        for line in lines:
            # Strip any leading/trailing whitespace and split by spaces
            elements = line.strip().split()
            
            # Convert the elements to the appropriate type (e.g., float)
            elements = [float(e) if '.' in e else int(e) for e in elements]
            
            # Append the processed elements to the data list
            data.append(elements)
    
    return data

parser = []
for file in txt_list:
    data = extract_data(file)
    file_name = [os.path.basename(file)]
    print(file_name)
    print(data)
    parser.append([file_name, data])

# Create an empty list to store DataFrames
df_list = []

# Loop over the parser data
for file_data in parser:
    file_name = file_data[0]
    data = file_data[1]
    
    # Create a DataFrame for the current file
    df = pd.DataFrame(data, columns=['label', 'x_center', 'y_center', 'width', 'height'])
    
    # Add the file name column, repeating the file name for each row
    df['file_name'] = file_name * len(df)
    
    # Append the DataFrame to the list
    df_list.append(df)

# Concatenate all DataFrames into a single DataFrame
result_df = pd.concat(df_list, ignore_index=True)

# Reorder columns to have file name first
result_df = result_df[['file_name', 'label', 'x_center', 'y_center', 'width', 'height']]

print(result_df)

def augment_data(image,image_name,df):
    """ Peforming vertical and horizontal flip"""
    bboxes = []
   
    image = np.array(image)

    group = df[df['file_name'] == image_name]
        # Get bounding boxes on coordinates of the image
    for index, row in group.iterrows():
            
            bboxes.append([row['x_center'], row['y_center'], row['width'], row['height'],row['label']])

    bbox_params = A.BboxParams(format='yolo', min_visibility=0.7)
    
    transform = A.Compose([
                               A.HorizontalFlip(p=0.5),
                               A.VerticalFlip(p=0.5)
                               #A.RandomBrightnessContrast(p=0.5),
                               #A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.5)
                               ],  
                               bbox_params = bbox_params)
    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bbox = transformed['bboxes']

    return(transformed_image, transformed_bbox)

def write_bboxes_to_txt(bboxes, filepath):
    with open(filepath, 'w') as file:
        for bbox in bboxes:
            x_center, y_center, width, height, label = bbox
            line = f"{label} {x_center} {y_center} {width} {height}\n"
            file.write(line)


def draw_yolo(image, labels, file_name):
    """ function to draw bounding box in the augmented image"""
    labelled_folder = 'datasets/train/augmented_labels'
    os.makedirs(labelled_folder, exist_ok=True)

    H, W = image.shape[:2]

    for label in labels:
        yolo_normalized = label[0:-1]
        box_voc = pbx.convert_bbox(tuple(yolo_normalized), from_type="yolo", to_type="voc", image_size=(W, H))
        cv2.rectangle(image, (box_voc[0], box_voc[1]),
                      (box_voc[2], box_voc[3]), (0, 255, 0), 2)
    cv2.imwrite(f"{labelled_folder}/{file_name}", image)

original_folder = '/dataset/train/images'
images_names = [f for f in os.listdir(original_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
augmented_folder = '/datasets/train/augmented_images'

os.makedirs(augmented_folder, exist_ok=True)


for image_name in images_names:
    image_path = os.path.join(original_folder, image_name)
    image = cv2.imread(image_path)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    print("Processing the file : ",base_filename, "\n")
    new_image_name = base_filename + '_augment.jpg'
    new_txt_name = base_filename + '_augment.txt'

    output_path_image = os.path.join(augmented_folder, new_image_name)
    output_path_txt = os.path.join(augmented_folder, new_txt_name)
    
    transformed_image,transformed_bbox  = augment_data(image, (base_filename + ".txt"), result_df)
    cv2.imwrite(output_path_image, transformed_image)
    
    write_bboxes_to_txt(transformed_bbox, output_path_txt)
    draw_yolo(transformed_image, transformed_bbox, image_name)