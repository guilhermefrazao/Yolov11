from glob import glob
from natsort import natsorted

import os
import numpy as np
import pandas as pd

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


if __name__ == "__main__":

    TRAINING_SIZE = 100

    imagens_antigos = "datasets/roboflow/train/images"

    labels_antigos = "datasets/roboflow/train/labels"

    imagens_novos = "datasets/improving_model/train/images"

    labels_novos = "datasets/improving_model/train/labels"

    folder_images_organizado = natsorted(os.listdir(imagens_antigos))

    folder_images_organizado = folder_images_organizado[:TRAINING_SIZE]

    folder_labels_organizado = natsorted(os.listdir(labels_antigos))

    folder_labels_organizado = folder_labels_organizado[:TRAINING_SIZE]

    for file in folder_images_organizado:
        file_images_path = os.path.join(imagens_antigos, file)
        file_images_name = os.path.basename(file_images_path)
        new_file_images_path = os.path.join(imagens_novos, file_images_name)
        os.rename(file_images_path, new_file_images_path)

    for file in folder_labels_organizado:
        file_labels_path = os.path.join(labels_antigos, file)
        file_labels_name = os.path.basename(file_labels_path)
        new_file_labels_path = os.path.join(labels_novos, file_labels_name)
        os.rename(file_labels_path, new_file_labels_path)