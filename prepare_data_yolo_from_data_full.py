import os
import random
import shutil
from tqdm import tqdm

# === НАСТРОЙКИ ===
SRC_DIR = r"C:\Users\vapcbuild\PycharmProjects\neiroset_mango\data_full"   # где лежат подпапки классов
DST_DIR = r"C:\Users\vapcbuild\PycharmProjects\neiroset_mango\data_yolo"   # уже со структурой
IMG_EXT = (".jpg", ".jpeg", ".png")
MAX_PER_CLASS = 100         # по сколько фото брать с каждого класса
SPLIT_RATIOS = (0.7, 0.2, 0.1)   # train/val/test
SEED = 42
random.seed(SEED)

# === ПОЛУЧЕНИЕ ВСЕХ КЛАССОВ (подпапок) ===
classes = [d for d in os.listdir(SRC_DIR) if os.path.isdir(os.path.join(SRC_DIR, d))]

print(f"\nНайдено классов: {len(classes)}")
for c in classes:
    print(" ", c)

all_images = []

# === ВЫБОРКА ПО 100 ИЗ КАЖДОГО КЛАССА ===
for cls in classes:
    cls_dir = os.path.join(SRC_DIR, cls)
    imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
            if f.lower().endswith(IMG_EXT)]
    random.shuffle(imgs)
    selected = imgs[:MAX_PER_CLASS] if len(imgs) > MAX_PER_CLASS else imgs
    all_images.extend(selected)

# === ПЕРЕМЕШАЕМ ВЕСЬ СОБРАННЫЙ НАБОР ===
random.shuffle(all_images)
print(f"\nВсего выбрано изображений: {len(all_images)}")

# === РАСЧЁТ ДОЛЕЙ ===
n_total = len(all_images)
n_train = int(n_total * SPLIT_RATIOS[0])
n_val = int(n_total * SPLIT_RATIOS[1])
splits = {
    "train": all_images[:n_train],
    "val": all_images[n_train:n_train + n_val],
    "test": all_images[n_train + n_val:]
}

# === КОПИРОВАНИЕ ===
for split_name, files in splits.items():
    img_dst = os.path.join(DST_DIR, split_name, "images")
    os.makedirs(img_dst, exist_ok=True)
    lbl_dst = os.path.join(DST_DIR, split_name, "labels")
    os.makedirs(lbl_dst, exist_ok=True)

    print(f"\n➡️ Копируем в {split_name} ({len(files)} файлов)")
    for src_path in tqdm(files, ncols=80):
        name = os.path.basename(src_path)
        dst_path = os.path.join(img_dst, name)
        shutil.copy2(src_path, dst_path)

# === ИТОГ ===
print("\n✅ Балансированный сплит завершён!")
print(f"Train: {len(splits['train'])} изображений")
print(f"Val:   {len(splits['val'])} изображений")
print(f"Test:  {len(splits['test'])} изображений")
print(f"\nПроверьте структуру в: {DST_DIR}")