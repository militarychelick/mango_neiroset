import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing

def main():
    # ===============================
    # УСТРОЙСТВО
    # ===============================
    device = torch.device("cpu")  # CPU-only
    print("DEVICE:", device)

    # ===============================
    # ПАРАМЕТРЫ
    # ===============================
    DATA_DIR = "../data_split"
    MODEL_PATH = "../mango_disease_model_pytorch.pth"
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS_PHASE1 = 10
    EPOCHS_PHASE2 = 20

    # ===============================
    # АУГМЕНТАЦИИ
    # ===============================
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

    # ===============================
    # ЗАГРУЗКА ДАТАСЕТОВ
    # ===============================
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)
    test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_transform)

    num_classes = len(train_ds.classes)
    print("Классы:", train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ===============================
    # МОДЕЛЬ
    # ===============================
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # ===============================
    # ФУНКЦИИ ОБУЧЕНИЯ
    # ===============================
    def train_epoch(loader, model, optimizer, epoch, total_epochs):
        model.train()
        total_loss, total_correct, total_incorrect = 0, 0, 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{total_epochs}", unit="batch",
                    bar_format="{l_bar}{bar:40}{r_bar}{bar:-10b}")

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

            pbar.set_postfix({
                "acc": f"{acc:.4f}",
                "correct": total_correct,
                "incorrect": total_incorrect,
                "loss": f"{total_loss / (total_correct + total_incorrect):.4f}"
            })

        return total_loss / len(loader.dataset), total_correct / len(loader.dataset), total_correct, total_incorrect

    def eval_epoch(loader, model):
        model.eval()
        total_loss, total_correct, total_incorrect = 0, 0, 0
        pbar = tqdm(loader, desc=f"Validation", unit="batch", bar_format="{l_bar}{bar:40}{r_bar}{bar:-10b}")
        with torch.no_grad():
            for imgs, labels in pbar:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * imgs.size(0)
                correct = (outputs.argmax(1) == labels).sum().item()
                total_correct += correct
                total_incorrect += imgs.size(0) - correct
                acc = total_correct / (total_correct + total_incorrect)

                pbar.set_postfix({
                    "acc": f"{acc:.4f}",
                    "correct": total_correct,
                    "incorrect": total_incorrect,
                    "loss": f"{total_loss / (total_correct + total_incorrect):.4f}"
                })

        return total_loss / len(loader.dataset), total_correct / len(loader.dataset), total_correct, total_incorrect

    def test_model(loader, model):
        loss, acc, correct, incorrect = eval_epoch(loader, model)
        print(f"\n=== Тестирование на тестовом наборе ===\naccuracy: {acc:.4f} - loss: {loss:.4f} - correct: {correct}, incorrect: {incorrect}")

    # ===============================
    # СПИСКИ ДЛЯ ГРАФИКОВ
    # ===============================
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # ===============================
    # ФАЗА 1 — ОБУЧЕНИЕ ГОЛОВЫ
    # ===============================
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    print(f"\n===== ФАЗА 1: обучение головы ({EPOCHS_PHASE1} эпох) =====")

    for epoch in range(EPOCHS_PHASE1):
        start_time = time.time()
        train_loss, train_acc, train_correct, train_incorrect = train_epoch(train_loader, model, optimizer, epoch, EPOCHS_PHASE1)
        val_loss, val_acc, val_correct, val_incorrect = eval_epoch(val_loader, model)
        epoch_time = int(time.time() - start_time)
        print(f"Epoch {epoch+1}/{EPOCHS_PHASE1} - train_acc: {train_acc:.4f}, train_loss: {train_loss:.4f}, correct: {train_correct}, incorrect: {train_incorrect} - val_acc: {val_acc:.4f}, val_loss: {val_loss:.4f}, val_correct: {val_correct}, val_incorrect: {val_incorrect} - time: {epoch_time}s")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    # ===============================
    # ФАЗА 2 — FINE-TUNING
    # ===============================
    for param in model.features.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    print(f"\n===== ФАЗА 2: fine-tuning ({EPOCHS_PHASE2} эпох) =====")

    for epoch in range(EPOCHS_PHASE2):
        start_time = time.time()
        train_loss, train_acc, train_correct, train_incorrect = train_epoch(train_loader, model, optimizer, epoch, EPOCHS_PHASE2)
        val_loss, val_acc, val_correct, val_incorrect = eval_epoch(val_loader, model)
        epoch_time = int(time.time() - start_time)
        print(f"Epoch {epoch+1}/{EPOCHS_PHASE2} - train_acc: {train_acc:.4f}, train_loss: {train_loss:.4f}, correct: {train_correct}, incorrect: {train_incorrect} - val_acc: {val_acc:.4f}, val_loss: {val_loss:.4f}, val_correct: {val_correct}, val_incorrect: {val_incorrect} - time: {epoch_time}s")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    # ===============================
    # СОХРАНЕНИЕ МОДЕЛИ
    # ===============================
    torch.save({
        "model_state": model.state_dict(),
        "classes": train_ds.classes
    }, MODEL_PATH)
    print("\nМодель сохранена:", MODEL_PATH)

    # ===============================
    # ТЕСТИРОВАНИЕ
    # ===============================
    test_model(test_loader, model)

    # ===============================
    # СТРОИМ ГРАФИКИ ОБУЧЕНИЯ
    # ===============================
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
    print("\nГрафик обучения сохранён: training_plot.png")
    plt.show()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()