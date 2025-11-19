import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import os
# =================
#  Bagian load data
# =================
csv_path = "dataset/Coffee Bean.csv"
df = pd.read_csv(csv_path)
df.head()
# =================
#  misah train test
# =================
train_df = df[df["data set"] == "train"]
test_df  = df[df["data set"] == "test"]
print("Train:", len(train_df))
print("Test :", len(test_df))
# =================
#  func load data
# =================
IMG_SIZE = 128

def load_image(path):
    img = load_img("dataset/" + path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img) / 255.0
    return img

def load_dataset(df):
    images = []
    labels = []

    for i, row in df.iterrows():
        img = load_image(row["filepaths"])
        images.append(img)
        labels.append(row["class index"])

    return np.array(images), np.array(labels)

x_train, y_train = load_dataset(train_df)
x_test,  y_test  = load_dataset(test_df)

x_train.shape, x_test.shape
# =======================
#  Model CNN + CIELAB(b*)
# =======================
def B_star_block(inputs, units=128, dropout=0.3):
    x = layers.Dense(units, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    return x

def create_hybrid_model(num_classes=4):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # --- CNN feature extractor ---
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)

    # --- B* Block ---
    x = B_star_block(x, 256, 0.4)
    x = B_star_block(x, 128, 0.3)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model

model = create_hybrid_model()
save_dir = "models/hybrid"
os.makedirs(save_dir, exist_ok=True)

checkpoint_path = os.path.join(save_dir, "hybrid_best_model.keras")
# callback
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]
model.summary()
# =================
#  Train model
# =================
history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)
# =================
# Simpan model
# =================
final_model_path = os.path.join(save_dir, "hybrid_final_model.keras")
model.save(final_model_path)

print(f"Model final disimpan di: {final_model_path}")
print(f"Model terbaik disimpan di: {checkpoint_path}")

# =================
#  Grafik Training
# =================
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.legend(["train","val"])

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(["train","val"])

plt.show()
# =================
#  Confusion Matrix
# =================
y_pred = np.argmax(model.predict(x_test), axis=1)

labels = ["Dark", "Green", "Light", "Medium"]

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

plt.figure(figsize=(6,6))
disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix")
plt.show()
# ============================
#  Evaluasi Model di Test Set
# ============================
from sklearn.metrics import classification_report, accuracy_score

print("\n==============================")
print("🔥 Evaluasi Model (Test Set)")
print("==============================")

# Prediksi kelas
y_pred = np.argmax(model.predict(x_test), axis=1)

# Akurasi
test_acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy : {test_acc:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Dark", "Green", "Light", "Medium"]))

# Confusion Matrix (simpan sebagai gambar juga)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.title("Confusion Matrix (Hybrid CNN + B*)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("hybrid_confusion_matrix.png")
print("\nConfusion Matrix disimpan → hybrid_confusion_matrix.png")
plt.show()


