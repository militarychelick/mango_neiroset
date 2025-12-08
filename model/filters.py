import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Загружаем модель заболевания
MODEL_PATH = os.path.join("model", "mango_disease_model.h5")
disease_model = load_model(MODEL_PATH)

# IMAGENET-сеть для определения формы листа
# Лёгкая, быстрая, не ломает цвета
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2

leaf_detector = MobileNetV2(weights="imagenet", include_top=True)


# ========== ПРЕДСКАЗАНИЕ БОЛЕЗНИ ==========
def predict_disease(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = disease_model.predict(img)[0]
    class_idx = np.argmax(preds)
    confidence = preds[class_idx]

    return class_idx, float(confidence)


# ========== ФИЛЬТР ЛИСТА МАНГО ==========
def is_mango_leaf(img_path):
    """
    Фильтр, который проверяет:
    ✔ есть ли ЛИСТ по ImageNet
    ✔ форма вытянутая (коэффициент формы)
    ✔ зелёная/желтоватая гамма
    """

    try:
        img = cv2.imread(img_path)
        if img is None:
            return False

        # ---- 1. Проверка на лист по MobileNetV2 ----
        img2 = cv2.resize(img, (224, 224))
        x = preprocess_input(np.expand_dims(img2.astype("float32"), axis=0))
        preds = leaf_detector.predict(x)

        # получаем top-1 класс
        decoded = preds[0]
        top_idx = np.argmax(decoded)

        # номера классов ImageNet, относящиеся к листьям
        LEAF_CLASSES = {
            989,  # leaf beet
            984,  # rapeseed leaf
            985,  # seaweed leaf
            988,  # cabbage leaf
            966,  # plant
        }

        if top_idx not in LEAF_CLASSES:
            return False  # не лист, стоп

        # ---- 2. Проверка формы листа ----
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)

        # Манго-лист всегда сильно вытянут (AR 2.0–6.0)
        if aspect_ratio < 1.8:
            return False

        # ---- 3. Цветовой фильтр (мягкий!) ----
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        green_pixels = np.sum((h > 20) & (h < 90) & (s > 40))
        percent_green = green_pixels / (img.shape[0] * img.shape[1])

        # Манго-лист обычно ≥20% зелёный
        if percent_green < 0.15:
            return False

        return True

    except Exception:
        return False