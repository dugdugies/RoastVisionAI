import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# =========================================================
# 1️⃣ KONFIGURASI DASAR
# =========================================================
COLOR_MODE = "YCbCr"  # 🔄 Ubah ke "RGB" kalau mau dataset asli
EPOCHS_PHASE1 = 100
EPOCHS_PHASE2 = 100
BATCH_SIZE = 16
IMG_SIZE = (224, 224)
MODEL_DIR = "models"

# Tentukan dataset berdasarkan COLOR_MODE
if COLOR_MODE == "YCbCr":
    BASE_DIR = "dataset/preprocessed"
else:
    BASE_DIR = "dataset/raw"

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"\n🎨 Mode Warna Aktif: {COLOR_MODE}")
print(f"📂 Dataset digunakan: {BASE_DIR}")

# =========================================================
# 2️⃣ IMAGE DATA GENERATOR
# =========================================================
if COLOR_MODE == "YCbCr":
    # Dataset hasil preprocessing → gunakan normalisasi umum
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.15,
        brightness_range=[0.85, 1.15],
        horizontal_flip=True,
        vertical_flip=False,
        shear_range=0.1,
        width_shift_range=0.15,
        height_shift_range=0.15,
        fill_mode='nearest',
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

else:
    # Dataset RGB asli → gunakan preprocess_input bawaan MobileNetV2
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        zoom_range=0.15,
        brightness_range=[0.85, 1.15],
        horizontal_flip=True,
        vertical_flip=False,
        shear_range=0.1,
        width_shift_range=0.15,
        height_shift_range=0.15,
        fill_mode='nearest',
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

# =========================================================
# 3️⃣ LOAD DATASET
# =========================================================
train_data = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42
)

val_data = val_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42
)

print("\n🔎 Kelas terdeteksi:", train_data.class_indices)
print(f"📊 Training samples: {train_data.samples}")
print(f"📊 Validation samples: {val_data.samples}")

# =========================================================
# 4️⃣ CLASS WEIGHTS
# =========================================================
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weight_dict = dict(enumerate(class_weights))
print("\n⚖️ Class Weights:", class_weight_dict)

# =========================================================
# 5️⃣ BANGUN MODEL
# =========================================================
print("\n🔧 Loading MobileNetV2...")
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = Sequential([
    Input(shape=(224, 224, 3)),
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
], name=f"CoffeeRoast_{COLOR_MODE}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n📋 Model Summary:")
model.summary()

# =========================================================
# 6️⃣ CALLBACKS
# =========================================================
checkpoint_phase1 = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, f"coffee_phase1_best_{COLOR_MODE}.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# =========================================================
# 7️⃣ PHASE 1 TRAINING (Frozen Base)
# =========================================================
print("\n" + "="*60)
print(f"🚀 PHASE 1 Training ({COLOR_MODE}) - Base Frozen")
print("="*60)

history_phase1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_PHASE1,
    class_weight=class_weight_dict,
    callbacks=[checkpoint_phase1, early_stop, reduce_lr],
    verbose=1
)

# =========================================================
# 8️⃣ PHASE 2 FINE-TUNING
# =========================================================
print("\n" + "="*60)
print(f"🔥 PHASE 2 Fine-Tuning ({COLOR_MODE})")
print("="*60)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint_phase2 = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, f"coffee_phase2_best_{COLOR_MODE}.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

history_phase2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_PHASE2,
    class_weight=class_weight_dict,
    callbacks=[checkpoint_phase2, early_stop, reduce_lr],
    verbose=1
)

# =========================================================
# 9️⃣ SIMPAN MODEL FINAL
# =========================================================
final_model_path = os.path.join(MODEL_DIR, f"CoffeeRoast_MobileNetV2_{COLOR_MODE}_final.keras")
model.save(final_model_path)

print("\n" + "="*60)
print("✅ Training Selesai!")
print("="*60)
print(f"📁 Model Phase 1: {MODEL_DIR}/coffee_phase1_best_{COLOR_MODE}.keras")
print(f"📁 Model Phase 2: {MODEL_DIR}/coffee_phase2_best_{COLOR_MODE}.keras")
print(f"📁 Model Final: {final_model_path}")

# =========================================================
# 🔟 EVALUASI
# =========================================================
print("\n" + "="*60)
print(f"📊 Evaluasi Model ({COLOR_MODE})")
print("="*60)

val_loss, val_acc = model.evaluate(val_data, verbose=0)
print(f"✅ Validation Accuracy: {val_acc*100:.2f}%")
print(f"❌ Validation Loss: {val_loss:.4f}")

print("\n🔍 Generating predictions...")
val_predictions, val_labels = [], []
for images, labels in val_data:
    preds = model.predict(images, verbose=0)
    val_predictions.extend(np.argmax(preds, axis=1))
    val_labels.extend(np.argmax(labels, axis=1))
    if len(val_predictions) >= val_data.samples:
        break

val_predictions = val_predictions[:val_data.samples]
val_labels = val_labels[:val_data.samples]

class_names = list(train_data.class_indices.keys())
cm = confusion_matrix(val_labels, val_predictions)

print("\n📋 Classification Report:")
print(classification_report(val_labels, val_predictions, target_names=class_names))

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix - Coffee Roast ({COLOR_MODE})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, f'confusion_matrix_{COLOR_MODE}.png'))

print("\n✨ Selesai!")
