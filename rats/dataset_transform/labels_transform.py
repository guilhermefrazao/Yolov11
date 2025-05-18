import os
from glob import glob


pasta_labels = "datasets/roboflow/test/labels"

arquivos_txt = glob(os.path.join(pasta_labels, '*.txt'))

for caminho_arquivo in arquivos_txt:
    novas_linhas = []
    
    with open(caminho_arquivo, 'r') as f:
        linhas = f.readlines()
        for linha in linhas:
            elementos = linha.strip().split()
            if len(elementos) > 0 and elementos[0] == '1':
                elementos[0] = '0'
            novas_linhas.append(' '.join(elementos) + '\n')
    
    with open(caminho_arquivo, 'w') as f:
        f.writelines(novas_linhas)

print("Conversão concluída.")


