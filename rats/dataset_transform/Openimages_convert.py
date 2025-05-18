import pandas as pd
import os
from PIL import Image

df = pd.read_csv("datasets/open-images-v7/train/labels/detections.csv")

IMAGES_DIR = "datasets/open-images-v7/train/data"

OUTPUT_TXT = "datasets/augmentation/augmented_labels_test/k.txt"

with open(OUTPUT_TXT, "w") as out:

    for _, row in df.iterrows():
        img_id    = "0a1dcafd3e7dc14d"
        xmin_norm = row["XMin"]
        ymin_norm = row["YMin"]
        xmax_norm = row["XMax"]
        ymax_norm = row["YMax"]

        #Abra a imagem só para pegar width/height
        img_path = os.path.join(IMAGES_DIR, f"{img_id}.jpg")

        try:
            with Image.open(img_path) as im:
                im.resize((640, 480))
                width, height = im.size
        except FileNotFoundError:
            print(f"⚠️  Imagem não encontrada: {img_path}, pulando…")
            continue

        #Converter normalizado → pixels
        startX = int(xmin_norm * width)
        startY = int(ymin_norm * height)
        endX   = int(xmax_norm * width)
        endY   = int(ymax_norm * height)

        #Escreve uma linha no TXT
        out.write(f"{img_id},{startX},{startY},{endX},{endY} \n")
