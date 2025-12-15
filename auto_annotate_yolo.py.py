from ultralytics import YOLO
from pathlib import Path
import os
from tqdm import tqdm

# === Настройки ===
BEST_WEIGHTS = r"C:\Users\vapcbuild\PycharmProjects\neiroset_mango\runs\detect\mango_yolo3\weights\best.pt"
DATA_DIR = r"C:\Users\vapcbuild\PycharmProjects\neiroset_mango\data_yolo"
CONF = 0.15    # чуть мягче, чтобы не терять листья
IOU = 0.5

# === Загрузка обученной модели ===
model = YOLO(BEST_WEIGHTS)

# === Функция авто‑разметки ===
def auto_annotate(split):
    img_dir = Path(DATA_DIR) / split / "images"
    lbl_dir = Path(DATA_DIR) / split / "labels"
    os.makedirs(lbl_dir, exist_ok=True)

    imgs = [f for f in img_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    print(f"\n🔹 {split.upper()}: найдено {len(imgs)} изображений")

    for p in tqdm(imgs, desc=f"{split}"):
        results = model.predict(source=str(p), conf=CONF, iou=IOU, save=False, verbose=False)
        # если YOLO ничего не нашла — пропускаем
        if not results or not len(results[0].boxes):
            continue

        boxes = results[0].boxes.xywhn.cpu().numpy()  # нормализованные координаты
        label_path = lbl_dir / (p.stem + ".txt")

        with open(label_path, "w") as f:
            for (x, y, w, h) in boxes:
                f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    print(f"✅ {split} — созданы .txt координат для {len(imgs)} фото")

# === Запуск для train / val / test ===
for part in ["train", "val", "test"]:
    auto_annotate(part)

print("\nГотово! Все обновлённые .txt сохранены в папках labels/.")