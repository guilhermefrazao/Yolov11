from glob import glob
from natsort import natsorted

import numpy as np
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import os

    
def extract_data(file_path): # Adiciona elementos extraidos de um arquivo passado como argumento e armazena na lista "data"
    data = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            elements = line.strip().split()

            elements = [float(e) if '.' in e else int(e) for e in elements]

            data.append(elements)
    
    return data

def labels_dataframe(txt_list): #Cria um dataframe com o nome do arquivo e suas labels
    parser = []
    df_list = []


    for file in txt_list:
        data = extract_data(file)
        file_name = [os.path.basename(file)]
        parser.append([file_name, data])

    for file_data in parser:
        file_name = file_data[0]
        data = file_data[1]
        
        if len(data) > 1:
            for dados in data:
                dataframe = [data[0]]
                df = pd.DataFrame(dataframe, columns=['classe', 'x_center', 'y_center', 'width', 'height'])
        
        else:
            df = pd.DataFrame(data, columns=['classe', 'x_center', 'y_center', 'width', 'height'])
        
        df['file_name'] = file_name * len(df)
        
        df_list.append(df)


    result_df = pd.concat(df_list, ignore_index=True)

    result_df = result_df[['file_name', 'classe', 'x_center', 'y_center', 'width', 'height']]

    print("result_df: ", result_df)

    return result_df

def augment_data(image,image_name,df): #Realiza o data_augmentation e cria os labels
    """ Peforming vertical and horizontal flip"""
    bboxes = []
   
    image = np.array(image)

    group = df[df['file_name'] == image_name]

    for index, row in group.iterrows():
            bboxes.append([row['x_center'], row['y_center'], row['width'], row['height'],row['classe']])

    bbox_params = A.BboxParams(format='yolo', min_visibility=0.7)
    
    transform = A.Compose([
                               A.HorizontalFlip(p=0.5),
                               A.VerticalFlip(p=0.5),
                               A.Defocus(radius=(4, 8), alias_blur=(0.2, 0.4), always_apply=True),
                               A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1.0),
                               A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                               ],  
                               bbox_params = bbox_params)
    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bbox = transformed['bboxes']

    print("Bounding_box_augmented: ", transformed_bbox)

    return(transformed_image, transformed_bbox)

def write_bboxes_to_txt(bboxes, filepath): #Escreve as bounding boxes no final 
    with open(filepath, 'w') as file:
        for bbox in bboxes:
            x_center, y_center, width, height, classe = bbox
            line = f"{classe} {x_center} {y_center} {width} {height}\n"
            file.write(line)

if __name__ == "__main__":

    labels_dir = "datasets/train/labels"
    original_folder = 'datasets/train/images'

    #lables_dir = 'datasets/train/labels_open_images'
    #original_folder = 'datasets/train/images_open_images'
    
    augmented_folder_images = 'datasets/augmentation/augmented_images'
    augmented_folder_txt = 'datasets/augmentation/augmented_labels'
    images_amount = 0

    txt_list = glob(labels_dir + '/**/*.txt', recursive = True)

    result_df = labels_dataframe(txt_list)

    folder_organizado = natsorted(os.listdir(original_folder))

    images_names = [f for f in folder_organizado if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_name in images_names:
        print("Processing the file : ",image_name , "\n")
        images_amount += 1
        print("images_amount:", images_amount)
        
        image_path = os.path.join(original_folder, image_name)
        image = cv2.imread(image_path)
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        print("Processing the file : ",base_filename , "\n")
        new_image_names = base_filename + '_augment.jpg'
        new_txt_name = base_filename + '_augment.txt'

        output_path_image = os.path.join(augmented_folder_images, new_image_names)
        output_path_txt = os.path.join(augmented_folder_txt, new_txt_name)
        
        transformed_image, transformed_bbox  = augment_data(image, (base_filename + ".txt"), result_df)

        cv2.imwrite(output_path_image, transformed_image)
        
        write_bboxes_to_txt(transformed_bbox, output_path_txt)
        