import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# === KONFIGURASI ===
DATASET_DIR = "dataset/raw"   # ubah ke folder dataset kamu
EXTENSIONS = ('.jpg', '.jpeg', '.png')

# === AMBIL NILAI LIGHTNESS SETIAP GAMBAR ===
lightness_values = []

for root, _, files in os.walk(DATASET_DIR):
    for file in files:
        if file.lower().endswith(EXTENSIONS):
            path = os.path.join(root, file)
            img = cv2.imread(path)

            if img is None:
                continue

            # Konversi ke CIELAB
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            L, A, B = cv2.split(lab)

            # Ambil nilai rata-rata L (0–255)
            lightness_mean = np.mean(L)
            lightness_values.append(lightness_mean)

# === CEK HASIL DASAR ===
print(f"Total gambar terbaca: {len(lightness_values)}")
print(f"Rata-rata Lightness keseluruhan: {np.mean(lightness_values):.2f}")

# === PLOT HISTOGRAM ===
plt.figure(figsize=(8, 5))
plt.hist(lightness_values, bins=30, edgecolor='black')
plt.title("Distribusi Nilai Lightness (L*) dari Dataset")
plt.xlabel("Rata-rata L (0–255)")
plt.ylabel("Jumlah Gambar")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
