import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
import cv2
import numpy as np
from PIL import Image

# ===============================
# ФУНКЦИЯ ОБРЕЗКИ ЛИСТА
# ===============================
def crop_leaf(img_path):
    """
    Возвращает PIL.Image с обрезанным листом по зеленому цвету.
    Если не найден лист, возвращает оригинал.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Не удалось открыть изображение")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Диапазон зеленого листьев манго
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Морфология для очистки маски
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return Image.open(img_path).convert("RGB")

    # Берем самый большой контур
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    cropped = img[y:y+h, x:x+w]

    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
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
    device = torch.device("cpu")  # CPU-only
    print("DEVICE:", device)

    DATA_DIR = "../data_split"
    MODEL_PATH = "../mango_disease_model_pytorch.pth"
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS_PHASE1 = 10
    EPOCHS_PHASE2 = 20

    # Аугментации
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(25),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
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
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # ===== Функции обучения/валидации =====
    def train_epoch(loader, model, optimizer, epoch, total_epochs):
        model.train()
        total_loss, total_correct, total_incorrect = 0,0,0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}", unit="batch")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct = (outputs.argmax(1) == labels).sum().item()
            total_correct += correct
            total_incorrect += imgs.size(0) - correct
            acc = total_correct / (total_correct + total_incorrect)
            pbar.set_postfix({"acc": f"{acc:.4f}", "loss": f"{total_loss / (total_correct + total_incorrect):.4f}"})
        return total_loss / len(loader.dataset), total_correct / len(loader.dataset), total_correct, total_incorrect

    def eval_epoch(loader, model):
        model.eval()
        total_loss, total_correct, total_incorrect = 0,0,0
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
    train_losses, val_losses, train_accs, val_accs = [],[],[],[]

    # ФАЗА 1 - обучаем голову
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
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
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(EPOCHS_PHASE2):
        start = time.time()
        train_loss, train_acc, _, _ = train_epoch(train_loader, model, optimizer, epoch, EPOCHS_PHASE2)
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

    # Графики
    plt.figure(figsize=(10,5))
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

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()