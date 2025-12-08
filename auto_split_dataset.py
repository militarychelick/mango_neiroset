import os
import shutil
import random

# Папка с полным датасетом (в ней должны быть подпапки классов)
DATA_FULL = "data_full"

# Куда раскидываем
DATA_SPLIT = "data_split"

# Пропорции разбиения
TRAIN_P = 0.7
VAL_P = 0.15
TEST_P = 0.15


# Создание структуры папок
def make_dirs(classes):
    for split in ["train", "val", "test"]:
        for cls in classes:
            path = os.path.join(DATA_SPLIT, split, cls)
            os.makedirs(path, exist_ok=True)


def split_dataset():
    # Все классы = подпапки data_full
    classes = [
        d for d in os.listdir(DATA_FULL)
        if os.path.isdir(os.path.join(DATA_FULL, d))
    ]

    print("[INFO] Найдены классы:", classes)

    make_dirs(classes)

    for cls in classes:
        full_path = os.path.join(DATA_FULL, cls)
        files = [
            f for f in os.listdir(full_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(files)

        total = len(files)
        train_n = int(total * TRAIN_P)
        val_n = int(total * VAL_P)

        train_files = files[:train_n]
        val_files = files[train_n:train_n + val_n]
        test_files = files[train_n + val_n:]

        # Копируем
        for fname in train_files:
            shutil.copy(
                os.path.join(full_path, fname),
                os.path.join(DATA_SPLIT, "train", cls, fname)
            )

        for fname in val_files:
            shutil.copy(
                os.path.join(full_path, fname),
                os.path.join(DATA_SPLIT, "val", cls, fname)
            )

        for fname in test_files:
            shutil.copy(
                os.path.join(full_path, fname),
                os.path.join(DATA_SPLIT, "test", cls, fname)
            )

        print(f"[OK] {cls}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    print("\n[DONE] Разбиение завершено!")


if __name__ == "__main__":
    split_dataset()