import os
import shutil
import random
from tqdm import tqdm

DATA_FULL = "data_full"
DATA_SPLIT = "data_split"

LIMIT_PER_CLASS = 3000
TRAIN_P = 0.70
VAL_P = 0.15
TEST_P = 0.15

def make_dirs(classes):
    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(DATA_SPLIT, split, cls), exist_ok=True)

def split_dataset():
    classes = [
        d for d in os.listdir(DATA_FULL)
        if os.path.isdir(os.path.join(DATA_FULL, d))
    ]

    print("[INFO] Найдены классы:", classes)
    make_dirs(classes)

    for cls in classes:
        print(f"\n[PROCESS] Класс: {cls}")

        full_path = os.path.join(DATA_FULL, cls)
        files = [
            f for f in os.listdir(full_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(files)

        # === Ограничиваем ===
        files = files[:LIMIT_PER_CLASS]

        total = len(files)
        train_n = int(total * TRAIN_P)
        val_n = int(total * VAL_P)

        splits = [
            ("train", files[:train_n]),
            ("val", files[train_n:train_n + val_n]),
            ("test", files[train_n + val_n:])
        ]

        for split_name, split_files in splits:
            out_dir = os.path.join(DATA_SPLIT, split_name, cls)

            for fname in tqdm(split_files, desc=f"{cls} → {split_name}", ncols=90):
                shutil.copyfile(
                    os.path.join(full_path, fname),
                    os.path.join(out_dir, fname)
                )

        print(f"[OK] {cls}: train={train_n}, val={val_n}, test={total - train_n - val_n}")

    print("\n[DONE] Разбиение завершено!")

if __name__ == "__main__":
    split_dataset()
