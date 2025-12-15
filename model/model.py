import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np

# ======================================
# НАСТРОЙКИ И ПУТИ
# ======================================
IMG_SIZE = 260  # EfficientNet-B2 обучалась на ~260 пикселях
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "mango_disease_model_pytorch.pth")
YOLO_WEIGHTS = os.path.join(BASE_DIR, "yolo_best_final.pt")
SELF_LEARN_DIR = os.path.abspath(os.path.join(BASE_DIR, "../self_learn"))
os.makedirs(SELF_LEARN_DIR, exist_ok=True)

# ======================================
# КЛАССЫ БОЛЕЗНЕЙ
# ======================================
DISEASES_EN = [
    "Anthracnose", "Bacterial Canker", "Cutting Weevil",
    "Die Back", "Gall Midge", "Healthy",
    "Powdery Mildew", "Sooty Mould"
]

DISEASES_RU = [
    "Антракноз", "Бактериальный рак", "Долгоносик",
    "Отмирание ветвей", "Галлица", "Здоровый",
    "Мучнистая роса", "Сажа"
]

# ======================================
# ЗАГРУЗКА EFFICIENTNET
# ======================================
print("Загрузка PyTorch модели EfficientNet...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

model = models.efficientnet_b2(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(DISEASES_EN))
model.load_state_dict(checkpoint["model_state"])
model.eval()

# используем GPU/DirectML или CPU
try:
    import torch_directml
    device = torch_directml.device(0)
    model.to(device)
    print("Модель EfficientNet перенесена на GPU (DirectML)\n")
except Exception as e:
    device = torch.device("cpu")
    print("DirectML недоступен, используем CPU:", e, "\n")

# ======================================
# ЗАГРУЗКА YOLO-МОДЕЛИ
# ======================================
if not os.path.exists(YOLO_WEIGHTS):
    raise FileNotFoundError(f"Не найден YOLO-модель {YOLO_WEIGHTS}")

yolo_model = YOLO(YOLO_WEIGHTS)
print("YOLO-модель загружена:", YOLO_WEIGHTS, "\n")

# ======================================
# FALLBACK-МАСКА LAB + HSV
# ======================================
def crop_leaf_mask(img_path):
    """
    Fallback-обрезка листа: выделяет зелёный лист через LAB+HSV маску.
    """
    img = cv2.imread(img_path)
    if img is None:
        return Image.open(img_path).convert("RGB")

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mask_a = cv2.inRange(a, 90, 140)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    mask_s = cv2.inRange(s, 40, 255)

    mask = cv2.bitwise_and(mask_a, mask_s)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("⚠ Маска не выделила лист, возвращаю исходное изображение.")
        return Image.open(img_path).convert("RGB")

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Чуть увеличим рамку, чтобы не отрезать края листа
    pad = 10
    y1, y2 = max(0, y - pad), min(img.shape[0], y + h + pad)
    x1, x2 = max(0, x - pad), min(img.shape[1], x + w + pad)

    cropped = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped)

# ======================================
# YOLO + Fallback с проверкой площади рамки
# ======================================
def crop_leaf_yolo(img_path, conf=0.45, iou=0.45):
    """
    Находит лист через YOLO, fallback на маску при низкой уверенности
    или слишком малой рамке (<10% от всего изображения).
    """
    result = yolo_model.predict(img_path, conf=conf, iou=iou, verbose=False)

    # YOLO ничего не нашла
    if len(result) == 0 or len(result[0].boxes) == 0:
        print("⚠ YOLO не нашла лист, fallback‑маска.")
        return crop_leaf_mask(img_path)

    boxes = result[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    largest_idx = int(np.argmax(areas))

    conf_score = float(boxes.conf[largest_idx].cpu())
    x1, y1, x2, y2 = map(int, xyxy[largest_idx])

    img0 = cv2.imread(img_path)
    if img0 is None:
        return Image.open(img_path).convert("RGB")

    h_img, w_img = img0.shape[:2]
    bbox_area = (x2 - x1) * (y2 - y1)
    total_area = w_img * h_img
    ratio = bbox_area / total_area

    print(f"YOLO: conf={conf_score:.2f}, ratio={ratio:.2f}")

    # Fallback варианты
    if conf_score < 0.25:
        print("⚠ Низкий conf YOLO, перехожу к маске.")
        return crop_leaf_mask(img_path)

    if ratio < 0.10:
        print(f"⚠ Рамка меньше 10% кадра (ratio={ratio:.2f}), возвращаю исходный кадр.")
        return Image.open(img_path).convert("RGB")

    # Добавим небольшой паддинг
    pad = int(0.03 * max(h_img, w_img))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w_img, x2 + pad)
    y2 = min(h_img, y2 + pad)

    cropped = cv2.cvtColor(img0[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped)

# ======================================
# ТРАНСФОРМАЦИИ ДЛЯ PREDICT
# ======================================
predict_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_full_image(img_path: str):
    """
    Классифицирует весь кадр (без YOLO и маски) — используется при переанализе.
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print("Ошибка открытия файла:", e)
        return None, 0.0

    tensor = predict_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu()

    class_idx = int(torch.argmax(probs))
    confidence = float(probs[class_idx])
    print("FULL-image probabilities:", [round(x, 4) for x in probs.tolist()])
    return class_idx, confidence

# ======================================
# ПРЕДСКАЗАНИЕ КЛАССА
# ======================================
def predict_disease(img_path: str):
    """
    Полный пайплайн: YOLO-обрезка -> EfficientNet‑классификация.
    """
    img = crop_leaf_yolo(img_path, conf=0.45, iou=0.45)

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_cv)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    img_clahe = cv2.merge((l2, a, b))
    img = Image.fromarray(cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB))

    tensor = predict_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu()

    class_idx = int(torch.argmax(probs))
    confidence = float(probs[class_idx])
    print("Probabilities:", [round(x, 4) for x in probs.tolist()])
    return class_idx, confidence

# ======================================
# ДО-ОБУЧЕНИЕ (заглушка)
# ======================================
def retrain_model():
    print("[SELF-TRAIN] Дообучение пока отключено для PyTorch версии.")

# ======================================
# ПРОВЕРКА
# ======================================
if __name__ == "__main__":
    test_img = "test_leaf.jpg"
    idx, conf = predict_disease(test_img)
    print(f"\nПредсказание: {DISEASES_RU[idx]} ({conf * 100:.1f}% уверенности)")