from ultralytics import YOLO
import cv2
import os

# путь к весам YOLO
WEIGHTS = r"C:\Users\vapcbuild\PycharmProjects\neiroset_mango\model\yolo_best_final.pt"
# папка с изображениями
IMAGES_DIR = r"C:\Users\vapcbuild\PycharmProjects\neiroset_mango\data_yolo\val\images"

# куда сохранить результаты
SAVE_DIR = r"C:\Users\vapcbuild\PycharmProjects\neiroset_mango\yolo_preview"
os.makedirs(SAVE_DIR, exist_ok=True)

model = YOLO(WEIGHTS)

print("\n--- Предсказания YOLO (покажет изображения с рамками) ---\n")

for name in os.listdir(IMAGES_DIR):
    if not name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    path = os.path.join(IMAGES_DIR, name)
    results = model.predict(path, conf=0.25, save=False, verbose=False)  # инференс

    if len(results) and len(results[0].boxes):
        img = results[0].plot()        # отрисовка рамок и меток
        save_path = os.path.join(SAVE_DIR, name)
        cv2.imwrite(save_path, img[..., ::-1])
        print(f"Сохранено: {save_path}")
    else:
        print(f"{name} — рамки не найдены")

print(f"\nГотово! Проверяй результаты в папке: {SAVE_DIR}")