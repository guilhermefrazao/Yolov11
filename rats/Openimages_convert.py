import os
import pandas as pd

csv_file = "datasets/fiftyone/open-images-v7/train/labels/detections.csv" 
df = pd.read_csv(csv_file)

output_dir = "datasets/train/labels_open"
os.makedirs(output_dir, exist_ok=True)


class_map = {}  

# Percorrer o CSV e converter os rótulos
for _, row in df.iterrows():
    image_id = row["ImageID"]
    label = row["LabelName"]
    
    # Atribuir ID numérico à classe se não existir
    if label not in class_map:
        class_map[label] = len(class_map)

    class_id = class_map[label]

    # Converter coordenadas para formato YOLO
    x_min, x_max = row["XMin"], row["XMax"]
    y_min, y_max = row["YMin"], row["YMax"]
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    # Criar arquivo de anotação para a imagem
    label_file = os.path.join(output_dir, f"{image_id}.txt")
    with open(label_file, "a") as f:
        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Salvar mapeamento de classes
with open("classes.txt", "w") as f:
    for label, idx in class_map.items():
        f.write(f"{idx} {label}\n")

print(f"Conversão concluída! Anotações salvas em {output_dir}/")
