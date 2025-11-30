import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 1. KONFIGURASI
# =========================================================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_PHASE1 = 50
EPOCHS_PHASE2 = 50
MODEL_DIR = "models/mobilevnet/"

CSV_PATH = "dataset/Coffee Bean.csv"
DATASET_BASE = "dataset/"

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# 2. LOAD CSV
# =========================================================
df = pd.read_csv(CSV_PATH)

train_df = df[df["data set"] == "train"]
test_df  = df[df["data set"] == "test"]

print("Train samples:", len(train_df))
print("Test samples :", len(test_df))

num_classes = len(df["class index"].unique())

# =========================================================
# 3. Fungsi Load Dataset
# =========================================================
def load_image(path):
    full_path = os.path.join(DATASET_BASE, path)
    img = load_img(full_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img)
    img = preprocess_input(img)   # MobileNetV2 preprocessing
    return img

def load_dataset(df):
    images, labels = [], []
    for _, row in df.iterrows():
        img = load_image(row["filepaths"])
        images.append(img)
        labels.append(row["class index"])
    return np.array(images), np.array(labels)

print("\nLoading training images...")
x_train, y_train = load_dataset(train_df)

print("Loading testing images...")
x_test, y_test = load_dataset(test_df)

# One-hot encoding
y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_oh  = tf.keras.utils.to_categorical(y_test, num_classes)

# =========================================================
# 4. CLASS WEIGHT
# =========================================================
class_weights = compute_class_weight(
    "balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print("\nClass Weights:", class_weights)

# =========================================================
# 5. BANGUN MODEL
# =========================================================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(512, activation="relu"),
    Dropout(0.4),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================================================
# 6. CALLBACKS
# =========================================================
checkpoint_phase1 = ModelCheckpoint(
    os.path.join(MODEL_DIR, "phase1_best.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
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
# 7. PHASE 1 TRAINING (BASE FROZEN)
# =========================================================
history1 = model.fit(
    x_train, y_train_oh,
    validation_split=0.2,
    epochs=EPOCHS_PHASE1,
    class_weight=class_weights,
    callbacks=[checkpoint_phase1, early_stop, reduce_lr],
    batch_size=BATCH_SIZE,
    verbose=1
)

# =========================================================
# 8. PHASE 2 FINE TUNING
# =========================================================
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.00005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint_phase2 = ModelCheckpoint(
    os.path.join(MODEL_DIR, "phase2_best.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

history2 = model.fit(
    x_train, y_train_oh,
    validation_split=0.2,
    epochs=EPOCHS_PHASE2,
    class_weight=class_weights,
    callbacks=[checkpoint_phase2, early_stop, reduce_lr],
    batch_size=BATCH_SIZE,
    verbose=1
)

# =========================================================
# 9. SIMPAN MODEL FINAL
# =========================================================
final_path = os.path.join(MODEL_DIR, "CoffeeRoast_final.keras")
model.save(final_path)
print(f"\nModel saved to: {final_path}")

# =========================================================
# 10. EVALUASI
# =========================================================
print("\nEvaluating model...")
test_loss, test_acc = model.evaluate(x_test, y_test_oh, verbose=0)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss    : {test_loss:.4f}")

preds = np.argmax(model.predict(x_test), axis=1)

print("\nClassification Report:")
print(classification_report(y_test, preds))

cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (CSV)")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix_csv.png"))
print("\nDone.")
