import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===============================
# НАСТРОЙКИ
# ===============================
IMG_SIZE = (224, 224)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "mango_disease_model_pytorch.pth")

SELF_LEARN_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../self_learn")
)
os.makedirs(SELF_LEARN_DIR, exist_ok=True)

# ===============================
# КЛАССЫ
# ===============================
DISEASES_EN = [
    "Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back",
    "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould"
]

DISEASES_RU = [
    "Антракноз", "Бактериальный рак", "Долгоносик", "Отмирание ветвей",
    "Галлица", "Здоровый", "Мучнистая роса", "Сажа"
]

# ===============================
# ЗАГРУЗКА МОДЕЛИ
# ===============================
print("Загрузка модели...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")

MODEL = load_model(MODEL_PATH)

print("Модель загружена.")

# ===============================
# ПРЕДСКАЗАНИЕ
# ===============================
def predict_disease(img_path: str):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    X = image.img_to_array(img) / 255.0
    X = np.expand_dims(X, axis=0)

    preds = MODEL.predict(X)
    class_idx = int(np.argmax(preds))
    confidence = float(preds[0][class_idx])

    return class_idx, confidence


# ===============================
# ДО-ОБУЧЕНИЕ МОДЕЛИ
# ===============================
def retrain_model():
    global MODEL  # ← ВАЖНО: ставим САМОЕ ПЕРВОЕ в функции

    print("[SELF-TRAIN] Запуск дообучения...")

    self_data_path = SELF_LEARN_DIR

    # Проверка наличия новых данных
    classes = [
        d for d in os.listdir(self_data_path)
        if os.path.isdir(os.path.join(self_data_path, d))
    ]

    if not classes:
        print("[SELF-TRAIN] Новых данных нет.")
        return

    print(f"[SELF-TRAIN] Найдены данные классов: {classes}")

    datagen = ImageDataGenerator(rescale=1./255)

    generator = datagen.flow_from_directory(
        self_data_path,
        target_size=IMG_SIZE,
        batch_size=4,
        class_mode='categorical',
        shuffle=True
    )

    # обучаем 1 эпоху
    MODEL.fit(generator, epochs=1)

    # сохраняем обновлённую модель
    MODEL.save(MODEL_PATH)
    print("[SELF-TRAIN] Модель обновлена и сохранена.")

    # перезагружаем новую модель, чтобы бот работал на новых весах
    MODEL = load_model(MODEL_PATH)
    print("[SELF-TRAIN] Модель перезагружена.")