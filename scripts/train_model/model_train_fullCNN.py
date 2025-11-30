import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os


# ================================================================
# 1. Load Dataset dari Folder (Auto Train/Test Split 90:10)
# ================================================================

DATASET_PATH = "augmented_dataset"   # folder utama dataset
IMG_SIZE = 128
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.1,      # 10% untuk test set
    subset="training",
    seed=42
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
    validation_split=0.1,
    subset="validation",
    seed=42
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)
print(f"Train batches: {len(train_ds)}")
print(f"Test batches : {len(test_ds)}")


# ================================================================
# 2. Convert Dataset ke NumPy Arrays (biar cocok dengan CNN kamu)
# ================================================================

def ds_to_numpy(dataset):
    images, labels = [], []
    for batch_imgs, batch_labels in dataset:
        images.append(batch_imgs.numpy())
        labels.append(batch_labels.numpy())
    return np.vstack(images), np.concatenate(labels)

x_train, y_train = ds_to_numpy(train_ds)
x_test , y_test  = ds_to_numpy(test_ds)

# Normalisasi 0–1
x_train = x_train / 255.0
x_test  = x_test / 255.0

print("x_train:", x_train.shape)
print("x_test :", x_test.shape)


# ================================================================
# 3. FULL CNN Architecture
# ================================================================

def create_full_cnn(num_classes=num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
    x = layers.AveragePooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.AveragePooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = layers.AveragePooling2D((2,2))(x)

    x = layers.Conv2D(256, (3,3), activation="relu", padding="same")(x)
    x = layers.AveragePooling2D((2,2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


model = create_full_cnn()
model.summary()


# ================================================================
# 4. Folder Penyimpanan Model
# ================================================================

save_dir = "models/model_dagi/cobamodel"
os.makedirs(save_dir, exist_ok=True)

best_model_path = os.path.join(save_dir, "best_cnn_model.keras")


# ================================================================
# 5. Callbacks
# ================================================================

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]


# ================================================================
# 6. Training Model
# ================================================================

history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=40,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)


# ================================================================
# 7. Save Final Model
# ================================================================

final_model_path = os.path.join(save_dir, "cnn_final_model.keras")
model.save(final_model_path)

print("\nModel terbaik:", best_model_path)
print("Model final  :", final_model_path)


# ================================================================
# 8. Plot Loss & Accuracy
# ================================================================

plt.figure(figsize=(12,4))

# Loss
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.legend()

# Accuracy
plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.show()


# ================================================================
# 9. Confusion Matrix & Classification Report
# ================================================================

y_pred = np.argmax(model.predict(x_test), axis=1)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("Confusion Matrix - Full CNN")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))