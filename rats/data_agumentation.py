import keras
import tensorflow as tf
import os
from tensorflow import keras 
from keras import layers
from keras import preprocessing
import matplotlib.pyplot as plt

def save_images(batch, batch_idx):
    for i, img in enumerate(batch):
        # Converte o tensor para numpy array e remove a dimensão do batch
        img_array = img.numpy().astype("uint8")
        
        # Usar o índice do batch e da imagem para nomear o arquivo
        img_filename = f"../{augmentation_dir}/augmented_image_{batch_idx}_{i}.png"
        
        # Salvar a imagem usando Matplotlib
        plt.imsave(img_filename, img_array)

def save_labels(labels, batch_idx):
    for i, label in enumerate(labels):

        label_array = label.numpy().astype("float32")

        label_filename = f"../{labels_dir}/augmented_image_{batch_idx}_{i}.txt"

        files = "../datasets/train/labels_augmentation"

        with open(files, 'w') as file:
            file.writelines(f"{label_array[0]} {label_array[1]} {label_array[2]} {label_array[3]} {label_array[4]}")

ds = preprocessing.image_dataset_from_directory(directory="../datasets/train", image_size=(640, 480), batch_size=32, verbose=True)

augmentation_pipeline = layers.Pipeline([
    layers.AutoContrast(),
    layers.RandomContrast(factor=0.5),
    layers.RandomFlip(mode='horizontal'), # meaning, left-to-right
    layers.RandomFlip(mode='vertical'), # meaning, top-to-bottom
    layers.RandomRotation(factor=0.20),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
])

preprocessed_ds = ds.map(
    lambda x, y: (augmentation_pipeline(x),y),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)

augmentation_dir = "datasets/train/images_augmentation"
labels_dir = "datasets/train/labels_augmentation"


# Iterar sobre o dataset processado e salvar as imagens aumentadas
for batch_idx, (batch_images, batch_labels) in enumerate(preprocessed_ds):
    save_images(batch_images, batch_idx)
    save_labels(batch_labels, batch_idx)

print(f"Imagens processadas foram salvas no diretório: {augmentation_dir}")
print(f"Labels processadas foram salvas no diretório: {labels_dir}")