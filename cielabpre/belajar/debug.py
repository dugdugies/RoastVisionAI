import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ================================================
# Path konfigurasi
# ================================================
CSV_PATH = "dataset/Coffee Bean.csv"
DATASET_DIR = "dataset/"
MODEL_PATH = "models/hybrid/hybrid_final_model.keras"

# ================================================
# Load model
# ================================================
model = tf.keras.models.load_model(MODEL_PATH)

# ================================================
# Load test data
# ================================================
df = pd.read_csv(CSV_PATH)
df_test = df[df["data set"] == "test"].reset_index(drop=True)

X_test = []
y_test = []

for _, row in df_test.iterrows():
    img = tf.keras.utils.load_img(os.path.join(DATASET_DIR, row['filepaths']), target_size=(224, 224))
    img = tf.keras.utils.img_to_array(img) / 255.0
    X_test.append(img)
    y_test.append(row['class index'])

X_test = np.array(X_test)
y_test = np.array(y_test)

# Prediksi
y_pred = np.argmax(model.predict(X_test), axis=1)

# ================================================
# Confusion Matrix
# ================================================
labels = ["Dark", "Green", "Light", "Medium"]
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ================================================
# Per-Class Accuracy
# ================================================
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
acc_df = pd.DataFrame({"Class": labels, "Accuracy": per_class_accuracy})
print("\n=== Per-Class Accuracy ===")
print(acc_df)

plt.figure(figsize=(7,5))
plt.bar(labels, per_class_accuracy)
plt.ylim(0, 1)
plt.title("Per-Class Accuracy")
plt.ylabel("Accuracy")
plt.show()

# ================================================
# Classification Report
# ================================================
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=labels))
