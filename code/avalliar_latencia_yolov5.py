import time
import torch
from ultralytics import YOLO
import yaml

# Caminhos dos seus arquivos
model_path = r"C:\Users\jonas\OneDrive\Área de Trabalho\backup_yolov5n - 50 epocas - imagens 640\weights\best.pt"
dataset_yaml = r"C:\Users\jonas\Documents\Doutorado\dota_dataset\yolo-dataset\dota.yaml"

# Carregar modelo YOLOv5 com ultralytics
model = YOLO(model_path)

# Ler dataset YAML para pegar algumas imagens para teste
with open(dataset_yaml, 'r') as f:
    data = yaml.safe_load(f)

# Exemplo: pegar uma imagem de validação para teste (ou qualquer imagem)
# Aqui assumimos que o YAML tem o campo 'val' com caminho das imagens de validação
val_images_path = data.get('val', None)
if val_images_path is None:
    raise ValueError("Não foi possível encontrar caminho 'val' no arquivo YAML.")

# Para teste, carregue uma imagem (pode ser só 1 para medir latência)
# Se 'val' for um arquivo txt com lista de imagens, leia a lista e pegue uma imagem
import os

def get_sample_image(path_or_txt):
    if os.path.isfile(path_or_txt):
        # Se for txt, pega a primeira linha com caminho da imagem
        with open(path_or_txt, 'r') as f:
            lines = f.readlines()
            if not lines:
                raise ValueError("Arquivo de validação está vazio")
            return lines[0].strip()
    elif os.path.isdir(path_or_txt):
        # Se for pasta, pega qualquer imagem dentro dela (jpg/png)
        for fname in os.listdir(path_or_txt):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                return os.path.join(path_or_txt, fname)
        raise ValueError("Nenhuma imagem encontrada na pasta de validação")
    else:
        raise ValueError("Caminho de validação inválido")

sample_image = get_sample_image(val_images_path)
print(f"Imagem para teste: {sample_image}")

# Medir latência - inferência em loop para média estável
n_runs = 50
times = []

# Pré-processar imagem antes do loop para evitar tempo extra (opcional)
# Mas com ultralytics YOLO model() a imagem pode ser passada como path diretamente.

for _ in range(n_runs):
    start = time.time()
    results = model(sample_image)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    times.append(end - start)

avg_latency = sum(times) / n_runs
print(f"Latência média por inferência (segundos): {avg_latency:.4f}")
print(f"Latência média por inferência (milissegundos): {avg_latency*1000:.2f} ms")
