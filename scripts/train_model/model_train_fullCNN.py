import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os


# ==========================
# 1. Load CSV
# ==========================
csv_path = "dataset/Coffee Bean.csv"
df = pd.read_csv(csv_path)

train_df = df[df["data set"] == "train"]
test_df  = df[df["data set"] == "test"]

print("Train:", len(train_df))
print("Test :", len(test_df))


# ==========================
# 2. Load Dataset
# ==========================
IMG_SIZE = 128

def load_image(path):
    img = load_img("dataset/" + path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img) / 255.0
    return img

def load_dataset(df):
    images, labels = [], []

    for i, row in df.iterrows():
        img = load_image(row["filepaths"])
        images.append(img)
        labels.append(row["class index"])

    return np.array(images), np.array(labels)

x_train, y_train = load_dataset(train_df)
x_test, y_test   = load_dataset(test_df)


# ==========================
# 3. FULL CNN
# ==========================
def create_full_cnn(num_classes=4):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
    x = layers.AveragePooling2D(pool_size=(2,2))(x)

    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.AveragePooling2D(pool_size=(2,2))(x)

    x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = layers.AveragePooling2D(pool_size=(2,2))(x)

    x = layers.Conv2D(256, (3,3), activation="relu", padding="same")(x)
    x = layers.AveragePooling2D(pool_size=(2,2))(x)

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


# ==========================
# 4. Buat folder penyimpanan
# ==========================
save_dir = "models/model_dagi/cobamodel"
os.makedirs(save_dir, exist_ok=True)

checkpoint_path = os.path.join(save_dir, "best_cnn_model.keras")


# ==========================
# 5. Callbacks
# ==========================
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]


# ==========================
# 6. Train
# ==========================
history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=40,
    batch_size=32,
    callbacks=callbacks
)


# ==========================
# 7. Save Final Model
# ==========================
final_path = os.path.join(save_dir, "cnn_final_model3.keras")
model.save(final_path)

print("Model terbaik:", checkpoint_path)
print("Model final:  ", final_path)


# ==========================
# 8. Plot Training Graph
# ==========================
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


# ==========================
# 9. Confusion Matrix + Evaluasi
# ==========================
y_pred = np.argmax(model.predict(x_test), axis=1)

labels = ["Dark", "Green", "Light", "Medium"]

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - Full CNN")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=labels))
