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

# =========================================
# 1️⃣ Konfigurasi dasar
# =========================================
BASE_DIR = "dataset/raw"
MODEL_DIR = "models"
EPOCHS_PHASE1 = 100
EPOCHS_PHASE2 = 100 
BATCH_SIZE = 16  # Kurangi batch size untuk dataset kecil
IMG_SIZE = (224, 224)

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================
# 2️⃣ ImageDataGenerator - OPTIMIZED untuk Coffee
# =========================================
# PENTING: Kedua datagen perlu validation_split yang SAMA!
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

print("\n🔎 Kelas yang terdeteksi:", train_data.class_indices)
print(f"📊 Training samples: {train_data.samples}")
print(f"📊 Validation samples: {val_data.samples}")

# =========================================
# 3️⃣ Hitung Class Weights
# =========================================
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weight_dict = dict(enumerate(class_weights))
print("\n⚖️ Class weights:", class_weight_dict)

# =========================================
# 4️⃣ Load MobileNetV2 Pre-trained Model
# =========================================
print("\n🔧 Loading MobileNetV2 pre-trained model...")
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

# =========================================
# 5️⃣ Bangun Model dengan Custom Head
# =========================================
model = Sequential([
    Input(shape=(224, 224, 3)),
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(512, activation='relu'),  # Lebih besar
    Dropout(0.4),  # Sedikit turun
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
], name="CoffeeRoastVision_v2")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n📋 Model Summary:")
model.summary()

# =========================================
# 6️⃣ Callbacks
# =========================================
checkpoint_phase1 = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "coffee_phase1_best.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=12,  # Lebih toleran untuk dataset kecil
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

# =========================================
# 7️⃣ PHASE 1: Training dengan Frozen Base
# =========================================
print("\n" + "="*60)
print("🚀 PHASE 1: Training dengan Base Model Frozen")
print("="*60)

history_phase1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_PHASE1,
    class_weight=class_weight_dict,
    callbacks=[checkpoint_phase1, early_stop, reduce_lr],
    verbose=1
)

# =========================================
# 8️⃣ PHASE 2: Fine-tuning
# =========================================
print("\n" + "="*60)
print("🔥 PHASE 2: Fine-tuning Model")
print("="*60)

base_model.trainable = True

# Unfreeze hanya layer terakhir (lebih hati-hati untuk dataset kecil)
for layer in base_model.layers[:-30]:  # Freeze lebih banyak layer
    layer.trainable = False

print(f"✅ Trainable layers: {len([l for l in model.layers if l.trainable])}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),  # Learning rate lebih kecil
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint_phase2 = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "coffee_phase2_best.keras"),
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

# =========================================
# 9️⃣ Simpan Model Final
# =========================================
final_model_path = os.path.join(MODEL_DIR, "CoffeeRoast_MobileNetV2_final.keras")
model.save(final_model_path)

print("\n" + "="*60)
print("✅ Training Selesai!")
print("="*60)
print(f"📁 Model Phase 1: {MODEL_DIR}/coffee_phase1_best.keras")
print(f"📁 Model Phase 2: {MODEL_DIR}/coffee_phase2_best.keras")
print(f"📁 Model Final: {final_model_path}")

# =========================================
# 🔟 Evaluasi & Visualization
# =========================================
print("\n" + "="*60)
print("📊 Evaluasi Model")
print("="*60)

val_loss, val_accuracy = model.evaluate(val_data, verbose=0)
print(f"✅ Validation Accuracy: {val_accuracy*100:.2f}%")
print(f"❌ Validation Loss: {val_loss:.4f}")

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

val_predictions = []
val_labels = []

print("\n🔍 Generating predictions...")
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

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Coffee Roast Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))

print("\n✨ Selesai!")