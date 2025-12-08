import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ========= Ускорение TensorFlow =========
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ========= ПАРАМЕТРЫ =========
DATA_DIR = "../data_split"
MODEL_PATH = "../mango_disease_model.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 150

EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 20

# ========= Аугментации =========
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.25,
    brightness_range=[0.7, 1.3],
    channel_shift_range=25,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ========= Модель =========
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ========= Freeze base model =========
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ========= Callbacks =========
callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1),
    ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True)
]

# ========= Фаза 1 =========
print("\n===== ФАЗА 1: обучение головы (10 эпох) =====")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE1,
)

# ========= Разморозка =========
for layer in base_model.layers:
    layer.trainable = True

model.compile(
    optimizer=Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ========= Фаза 2 =========
print("\n===== ФАЗА 2: fine-tuning (20 эпох) =====")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE2,
)

# ========= Сохранение =========
model.save(MODEL_PATH)
print(f"\nМодель сохранена: {MODEL_PATH}")
