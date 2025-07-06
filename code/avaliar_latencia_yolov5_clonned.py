import sys
import pathlib
import time
import torch
import cv2
import numpy as np

# Ajustar caminho para importar yolov5 utils
repo_path = pathlib.Path(__file__).parent.resolve()
if str(repo_path) not in sys.path:
    sys.path.insert(0, str(repo_path))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# Caminho do modelo e imagem de teste
model_path = r"C:\Users\jonas\OneDrive\Área de Trabalho\backup_yolov5n - 50 epocas - imagens 640\weights\best.pt"
img_path = r"C:\Users\jonas\Documents\Doutorado\dota_dataset\yolo-dataset\datasets\images\test\P0024.png"

# Configurar device
device = select_device('0' if torch.cuda.is_available() else 'cpu')

# Carregar modelo
model = DetectMultiBackend(model_path, device=device, dnn=False)
model.eval()

# Função para pré-processar imagem no padrão YOLOv5 (redimensionar, normalizar)
def preprocess_image(img_path, img_size=640):
    img = cv2.imread(img_path)  # BGR
    assert img is not None, f'Imagem não encontrada {img_path}'

    # Pega tamanho original
    h0, w0 = img.shape[:2]

    # Redimensiona com padding mantendo proporção
    r = img_size / max(h0, w0)
    new_unpad = (int(w0 * r), int(h0 * r))
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Criar padding para img quadrada img_size x img_size
    dw = img_size - new_unpad[0]
    dh = img_size - new_unpad[1]
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)

    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))

    # Convert BGR para RGB
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)

    # Normalizar para [0,1]
    img_norm = img_rgb.astype(np.float32) / 255.0

    # HWC to CHW
    img_chw = np.transpose(img_norm, (2,0,1))

    # Adicionar batch dim
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device)

    return img_tensor

# Pré-processar a imagem
img = preprocess_image(img_path, img_size=640)

# Aquecimento GPU (opcional)
for _ in range(10):
    _ = model(img)

# Medir latência
n_runs = 50
times = []

for _ in range(n_runs):
    start = time.time()
    pred = model(img)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.time()
    times.append(end - start)

avg_latency = sum(times) / n_runs
print(f"Latência média por inferência: {avg_latency*1000:.2f} ms")
