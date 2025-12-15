import os
import time
import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2

# ===============================
# ФУНКЦИЯ ОБРЕЗКИ ЛИСТА
# ===============================
def crop_leaf(img_path):
    """
    Лучше выделяет лист (LAB + HSV маска).
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Не удалось открыть: {img_path}")
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
        return Image.open(img_path).convert("RGB")

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    cropped = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped)
# ===============================
# СОБСТВЕННЫЙ DATASET С ОБРЕЗКОЙ
# ===============================
class MangoDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.ds = datasets.ImageFolder(folder)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        path, label = self.ds.samples[idx]
        img = crop_leaf(path)
        if self.transform:
            img = self.transform(img)
        return img, label

# ===============================
# ОСНОВНАЯ ФУНКЦИЯ
# ===============================
def main():
     # Проверяем и подключаем DirectML / GPU / CPU
    try:
        import torch_directml
        import wmi  # библиотека для получения модели GPU
        # Печатаем список всех доступных DML устройств
        num_dev = torch_directml.device_count()
        print(f"\nНайдено DirectML устройств: {num_dev}")
        for i in range(num_dev):
            print(f"  {i}: {torch_directml.device(i)}")

        device = torch_directml.device(0)

        # Получаем название видеокарты через WMI (работает на Windows)
        try:
            comp = wmi.WMI()
            gpus = comp.Win32_VideoController()
            if gpus:
                    print(f"Модель видеокарты: {gpus[0].Name}")
        except Exception as e:
            print("Не удалось получить модель видеокарты:", e)

        print("DEVICE: GPU - DirectML\n")
    except Exception as e:
        device = torch.device("cpu")
        print("DEVICE: CPU (DirectML недоступен, причина:", e, ")\n")

    DATA_DIR = "../data_split"
    MODEL_PATH = "../mango_disease_model_pytorch.pth"
    BATCH_SIZE = 32
    EPOCHS_PHASE1 = 10
    EPOCHS_PHASE2 = 20

    IMG_SIZE = 260  # EfficientNet-B2 ожидает ~260px
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
        transforms.RandomAdjustSharpness(2.0),
        transforms.RandomAutocontrast(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Загрузка датасетов с кастомным Dataset
    train_ds = MangoDataset(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_ds = MangoDataset(os.path.join(DATA_DIR, "val"), transform=val_transform)
    test_ds = MangoDataset(os.path.join(DATA_DIR, "test"), transform=val_transform)

    num_classes = len(train_ds.ds.classes)
    print("Классы:", train_ds.ds.classes)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Модель
    model = models.efficientnet_b2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def mixup_data(x, y, alpha=0.4):
        """Смешивает два изображения и их метки."""
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    # ===== Функции обучения/валидации =====
    def train_epoch(loader, model, optimizer, epoch, total_epochs, scheduler=None):
        model.train()
        total_loss, total_correct, total_incorrect = 0, 0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{total_epochs}", unit="batch")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            imgs, y_a, y_b, lam = mixup_data(imgs, labels)
            outputs = model(imgs)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item() * imgs.size(0)
            correct = (outputs.argmax(1) == labels).sum().item()
            total_correct += correct
            total_incorrect += imgs.size(0) - correct
            acc = total_correct / (total_correct + total_incorrect)
            pbar.set_postfix({"acc": f"{acc:.4f}", "loss": f"{total_loss / (total_correct + total_incorrect):.4f}"})
        return total_loss / len(loader.dataset), total_correct / len(loader.dataset), total_correct, total_incorrect

    def eval_epoch(loader, model):
        model.eval()
        total_loss, total_correct, total_incorrect = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc="Validation", unit="batch"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * imgs.size(0)
                correct = (outputs.argmax(1) == labels).sum().item()
                total_correct += correct
                total_incorrect += imgs.size(0) - correct
        acc = total_correct / (total_correct + total_incorrect)
        return total_loss / len(loader.dataset), acc, total_correct, total_incorrect

    def test_model(loader, model):
        loss, acc, correct, incorrect = eval_epoch(loader, model)
        print(f"\n=== Тестирование ===\naccuracy: {acc:.4f} - loss: {loss:.4f} - correct: {correct}, incorrect: {incorrect}")

    # ===== Тренировка =====
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    # ФАЗА 1 - обучаем голову
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(EPOCHS_PHASE1):
        start = time.time()
        train_loss, train_acc, _, _ = train_epoch(train_loader, model, optimizer, epoch, EPOCHS_PHASE1)
        val_loss, val_acc, _, _ = eval_epoch(val_loader, model)
        epoch_time = int(time.time() - start)
        print(f"Epoch {epoch+1}/{EPOCHS_PHASE1} - train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, time: {epoch_time}s")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    # ФАЗА 2 - fine-tuning
    for param in model.features.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-5,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS_PHASE2,
    )
    for epoch in range(EPOCHS_PHASE2):
        start = time.time()
        train_loss, train_acc, _, _ = train_epoch(train_loader, model, optimizer, epoch, EPOCHS_PHASE2, scheduler)
        val_loss, val_acc, _, _ = eval_epoch(val_loader, model)
        epoch_time = int(time.time() - start)
        print(f"Epoch {epoch+1}/{EPOCHS_PHASE2} - train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, time: {epoch_time}s")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    # Сохраняем модель
    torch.save({
        "model_state": model.state_dict(),
        "classes": train_ds.ds.classes
    }, MODEL_PATH)
    print("\nМодель сохранена:", MODEL_PATH)

    # Тест
    test_model(test_loader, model)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    def plot_confusion_matrix(model, loader, classes):
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(1).cpu().numpy()
                y_true += labels.numpy().tolist()
                y_pred += preds.tolist()
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=classes)
        disp.plot(cmap="Greens", xticks_rotation=45)
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.show()

    plot_confusion_matrix(model, test_loader, train_ds.ds.classes)

    # Графики
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="train_loss", marker='o')
    plt.plot(val_losses, label="val_loss", marker='o')
    plt.plot(train_accs, label="train_acc", marker='o')
    plt.plot(val_accs, label="val_acc", marker='o')
    plt.title("Обучение модели")
    plt.xlabel("Эпохи")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_plot.png")
    plt.show()

# ==========================================================
# YOLO + EfficientNet (распознавание с IoU и обрезкой)
# ==========================================================
yolo_model_crop = YOLO("model/yolo_best_final.pt")

def crop_leaf_yolo(img_path, conf=0.25, iou=0.5):
    """
    Использует YOLO для нахождения листа и возвращает вырезанный фрагмент.
    conf — минимальная уверенность, iou — порог IoU.
    """
    result = yolo_model_crop.predict(img_path, conf=conf, iou=iou, verbose=False)
    if len(result) == 0 or len(result[0].boxes) == 0:
        # Ничего не найдено — возвращаем исходное изображение.
        print("⚠ YOLO не нашла лист, анализируем весь кадр.")
        return Image.open(img_path).convert("RGB")

    boxes = result[0].boxes
    if boxes is None or len(boxes.xyxy) == 0:
        print("⚠ YOLO не нашла лист, анализируем весь кадр.")
        return Image.open(img_path).convert("RGB")

    # вычисляем площади рамок шагом max((x2-x1)*(y2-y1))
    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    largest_idx = int(np.argmax(areas))
    x1, y1, x2, y2 = map(int, xyxy[largest_idx])
    img = cv2.imread(img_path)
    crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop)

# ==========================================================
# YOLO + EfficientNet: сравнение и визуализация
# ==========================================================
def compare_with_yolo():
    from PIL import ImageDraw
    import torch.nn.functional as F

    MODEL_PATH = "../mango_disease_model_pytorch.pth"
    TEST_DIR = "../data_split/test"
    SAVE_DIR = "./compare_pred"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Загружаем обученную EfficientNet
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes = checkpoint["classes"]

    eff_model = models.efficientnet_b2(weights=None)
    eff_model.classifier[1] = nn.Linear(eff_model.classifier[1].in_features, len(classes))
    eff_model.load_state_dict(checkpoint["model_state"])
    eff_model.eval()

    eff_tf = transforms.Compose([
        transforms.Resize(int(260 * 1.15)),
        transforms.CenterCrop(260),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def predict_eff(img_path, conf_yolo=0.25, iou_yolo=0.5):
        # 1️⃣ YOLO обрезает лист
        img = crop_leaf_yolo(img_path, conf=conf_yolo, iou=iou_yolo)

        # 2️⃣ EfficientNet классифицирует фрагмент
        t = eff_tf(img).unsqueeze(0)
        with torch.no_grad():
            out = eff_model(t)
            p = F.softmax(out, dim=1)[0]
            i = int(torch.argmax(p))
        return classes[i], float(p[i])

    # 2. Загружаем готовую YOLOv8n для рамок (без обучения)
    yolo = YOLO("yolov8n.pt")

    # 3. Пробегаем по тестовой выборке
    print("\n=== YOLO + EfficientNet visualize ===")
    for cls in os.listdir(TEST_DIR):
        cls_path = os.path.join(TEST_DIR, cls)
        if not os.path.isdir(cls_path):
            continue
        out_sub = os.path.join(SAVE_DIR, cls)
        os.makedirs(out_sub, exist_ok=True)

        for name in os.listdir(cls_path):
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(cls_path, name)

            # EfficientNet
            elabel, econf = predict_eff(path)

            # YOLO — просто подчёркиваем лист рамкой
            res = yolo.predict(path, conf=0.25, verbose=False)
            img_yolo = res[0].plot() if len(res) > 0 else cv2.imread(path)[..., ::-1]

            out_img = Image.fromarray(img_yolo)
            draw = ImageDraw.Draw(out_img)
            draw.text((10, 10), f"{elabel} ({econf*100:.1f}%)", fill="green")

            save_path = os.path.join(out_sub, name)
            out_img.save(save_path)
            print(f"{name} -> {elabel} ({econf:.2f})")

    print(f"\nВизуализированные результаты сохранены в: {SAVE_DIR}")

def train_yolo_detector():
    print("\n=== ФУНКЦИЯ train_yolo_detector() ВЫЗВАНА ===")
    print("Текущая рабочая директория:", os.getcwd())
    yaml_path = os.path.join(os.path.dirname(__file__), "yolo_data.yaml")
    print("Проверка файла данных:", yaml_path, "—", os.path.exists(yaml_path))

    print("\n=== Начинаем дообучение YOLO на disease классах ===")
    model_yolo = YOLO("yolov8n.pt")  # твой базовый файл

    results = model_yolo.train(
        data="C:/Users/vapcbuild/PycharmProjects/neiroset_mango/model/yolo_data.yaml",
        epochs=150,
        imgsz=640,
        batch=16,
        name="mango_yolo_best_adamW",
        optimizer="AdamW",
        lr0=0.001,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        translate=0.1,
        degrees=10,
        erasing=0.4,
        fliplr=0.5,
        iou=0.5,
        conf=0.15,
        workers=2,
        device='cpu'
    )

    print("\n=== Обучение YOLO завершено ===")
    print("Логи и веса сохранены в:", results.save_dir)

if __name__ == "__main__":
    # --- ФАЗА 1: EfficientNet ---
    # Пропускаем обучение EfficientNet, так как модель уже обучена
    print("\n✅ Найдена уже обученная EfficientNet: ../mango_disease_model_pytorch.pth")
    print("Пропускаем обучение и переходим сразу к YOLO.\n")

    # --- ФАЗА 2: YOLO ---
    print("\n=== Запускаем обучение YOLO ===\n")
    train_yolo_detector()

    # --- ФАЗА 3: сравнение и визуализация YOLO + EfficientNet ---
    print("\n=== Запускаем YOLO + EfficientNet визуализацию ===\n")
    compare_with_yolo()