import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from io import BytesIO

# ===== ПУТЬ К МОДЕЛИ =====
MODEL_PATH = os.path.join("model", "mango_disease_model_pytorch.pth")

# ===== ЗАГРУЗКА МОДЕЛИ =====

checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

# Инициализируем MobileNetV2 вручную
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

num_classes = len(checkpoint["classes"])

model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

model.load_state_dict(checkpoint["model_state"])
model.eval()

# список классов
CLASSES = checkpoint["classes"]

# ===== ПРЕОБРАЗОВАНИЕ ДЛЯ КАРТИНОК =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ===== ФУНКЦИЯ ПРЕДСКАЗАНИЯ =====
def predict_disease(img_source):
    """
    img_source — путь к файлу или BytesIO
    Возвращает (индекс класса, уверенность)
    """

    if isinstance(img_source, BytesIO):
        img = Image.open(img_source).convert("RGB")
    else:
        img = Image.open(str(img_source)).convert("RGB")

    img = transform(img)
    img = img.unsqueeze(0)  # добавляем batch

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1).numpy()[0]

    class_idx = int(probs.argmax())
    confidence = float(probs[class_idx])

    return class_idx, confidence