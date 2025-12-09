import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ======================================
# ПУТИ
# ======================================
IMG_SIZE = 224
MODEL_PATH = os.path.join(os.path.dirname(__file__), "mango_disease_model_pytorch.pth")
SELF_LEARN_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../self_learn")
)
os.makedirs(SELF_LEARN_DIR, exist_ok=True)

# ======================================
# КЛАССЫ
# ======================================
DISEASES_EN = [
    "Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back",
    "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould"
]

DISEASES_RU = [
    "Антракноз", "Бактериальный рак", "Долгоносик", "Отмирание ветвей",
    "Галлица", "Здоровый", "Мучнистая роса", "Сажа"
]

# ======================================
# ЗАГРУЗКА PyTorch-МОДЕЛИ
# ======================================
print("Загрузка PyTorch модели...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

# Создаем такую же сеть MobileNetV2
from torchvision import models
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(DISEASES_EN))
model.load_state_dict(checkpoint["model_state"])
model.eval()

print("Модель загружена.")

# ======================================
# Transform для предсказания
# ======================================
predict_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ======================================
# ФУНКЦИЯ ПРЕДСКАЗАНИЯ
# ======================================
def predict_disease(img_path: str):
    img = Image.open(img_path).convert("RGB")
    tensor = predict_tf(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    class_idx = int(torch.argmax(probs).item())
    confidence = float(probs[class_idx])

    return class_idx, confidence


# ======================================
# ДООБУЧЕНИЕ МОДЕЛИ (простая версия)
# ======================================
def retrain_model():
    print("[SELF-TRAIN] Дообучение пока отключено для PyTorch версии.")
