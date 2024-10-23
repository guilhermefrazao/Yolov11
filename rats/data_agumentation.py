import keras
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd

from glob import glob
from tensorflow import keras 
from keras import layers
from keras import preprocessing
from rats.augmentation_labels import augmentation_labels



def saving_files(preprocessed_ds, labels, augmentation_dir, labels_dir):
        for batch_idx, (batch_images) in enumerate(preprocessed_ds):

            for i, (img, lbl) in enumerate(zip(batch_images)):
                img_array = img.numpy().astype("uint8")
                img_filename = os.path.join(augmentation_dir, f"augmented_image_{batch_idx}_{i}.png")
                plt.imsave(img_filename, img_array)

                label_filename = os.path.join(labels_dir, f"augmented_image_{batch_idx}_{i}.txt")
                with open(label_filename, 'w') as file:
                    file.write(' '.join(map(str, lbl.numpy())))

def Augmentation(augmentation_dir, labels_dir):
    ds = preprocessing.image_dataset_from_directory(
        directory="datasets/train", image_size=(640, 480), batch_size=32, verbose=True)

    augmentation_pipeline = layers.Pipeline([
        layers.AutoContrast(),
        layers.RandomContrast(factor=0.5),
        layers.RandomBrightness(factor=0.3),
        layers.RandomFlip(mode='horizontal'), 
    ])

    preprocessed_ds = ds.map(
        lambda x, y: (augmentation_pipeline(x),y),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    augmentation_labels()

    saving_files(preprocessed_ds, augmentation_dir, labels_dir)

    return True

if __name__ == "__main__":
    augmentation_dir = "datasets/train/augmented_images"
    labels_dir = "datasets/train/augmented_labels"

    if Augmentation(augmentation_dir, labels_dir):
        print("Data augmentation concluída com sucesso!")
    else:
        print("Erro ao realizar data augmentation!")

