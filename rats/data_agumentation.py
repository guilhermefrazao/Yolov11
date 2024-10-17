import keras
import tensorflow as tf
import os
from tensorflow import keras 
from keras import layers
from keras import preprocessing
import matplotlib.pyplot as plt


def load_labels(labels_dir, image_filenames):
    labels = []
    for filename in image_filenames:
        label_filename = os.path.join(labels_dir, filename.replace(".jpg", ".txt"))  # Supondo que os arquivos de rótulo tenham o mesmo nome
        with open(label_filename, 'r') as f:
            labels.append(int(f.read().strip()))  # Supondo que o label seja um número
    return labels

def save_images(batch, batch_idx, augmentation_dir):
    for i, img in enumerate(batch):
        # Converte o tensor para numpy array e remove a dimensão do batch
        img_array = img.numpy().astype("uint8")
        
        # Usar o índice do batch e da imagem para nomear o arquivo
        img_filename = f"{augmentation_dir}/augmented_image_{batch_idx}_{i}.png"
        
        # Salvar a imagem usando Matplotlib
        plt.imsave(img_filename, img_array)

def save_labels(labels, batch_idx, labels_dir):
    for i, label in enumerate(labels):

        label_array = labels.numpy().astype("float32")
        
        label_filename = f"{labels_dir}/augmented_image_{batch_idx}_{i}.txt"

        with open(label_filename, 'w') as file:
            file.writelines(f"{label_array[0]} {label_array[1]} {label_array[2]} {label_array[3]} {label_array[4]}")

def saving_files(preprocessed_ds,augmentation_dir,labels_dir):
        for batch_idx, (batch_images, labels) in enumerate(preprocessed_ds):
            save_images(batch_images, batch_idx, augmentation_dir)
            save_labels(labels, batch_idx, labels_dir)

def Augmentation():
    ds = preprocessing.image_dataset_from_directory(
        directory="datasets/train", image_size=(640, 480), batch_size=32, verbose=True)

    augmentation_pipeline = layers.Pipeline([
        layers.AutoContrast(),
        layers.RandomContrast(factor=0.5),
        layers.RandomBrightness(factor=0.3),
        layers.RandomFlip(mode='horizontal'), # meaning, left-to-right
    ])

    preprocessed_ds = ds.map(
        lambda x, y: (augmentation_pipeline(x),y),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    augmentation_dir = "datasets/train/images"
    labels_dir = "datasets/train/labels"

    for images, labels in preprocessed_ds.take(1):  # 'take(1)' para ver apenas um batch
        print("Labels das imagens augmentadas:", labels.numpy())

    saving_files(preprocessed_ds, augmentation_dir, labels_dir)

    return True

if Augmentation():
    print("Augmentation done!")
else:
    print("Augmentation failed!")
