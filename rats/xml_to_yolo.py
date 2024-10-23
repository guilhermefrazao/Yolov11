import xml.etree.ElementTree as ET

tree = ET.parse("datasets/train_marcos/annotations.xml")
root = tree.getroot()
i = 0


for entry in root.findall("image"):
    bounding_box = entry.find("box")
    bounding_box = bounding_box.attrib
    image_width = float(entry.get("width"))
    image_height = float(entry.get("height"))

    xtl = float(bounding_box["xtl"])
    ytl = float(bounding_box["ytl"])
    xbr = float(bounding_box["xbr"])
    ybr = float(bounding_box["ybr"])

    x_direita = xtl / 1000
    y_direita = ytl / 1000
    x_esquerda = xbr / 1000
    y_esquerda = ybr  / 1000

    # Escrever os valores em um arquivo
    with open(f"datasets/train_marcos/labels/marcos_{i}.txt", "w") as file:
        file.write(f"0 {x_direita} {y_direita} {x_esquerda} {y_esquerda}\n")
        
    print("bounding_box:\n", bounding_box)

    i += 1

    
