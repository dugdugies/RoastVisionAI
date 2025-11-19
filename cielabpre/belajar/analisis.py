import cv2
import numpy as np
import os
from glob import glob

def analyze_folder(folder_path):
    image_paths = glob(os.path.join(folder_path, "*.png")) + glob(os.path.join(folder_path, "*.jpg"))
    if not image_paths:
        print(f"⚠️ Tidak ada gambar di folder: {folder_path}")
        return None

    L_values, A_values, B_values = [], [], []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Gagal baca gambar: {img_path}")
            continue

        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img_lab)

        Lmean = np.mean(l) * (100 / 255)
        Amean = np.mean(a) - 128
        Bmean = np.mean(b) - 128

        L_values.append(Lmean)
        A_values.append(Amean)
        B_values.append(Bmean)

    # hitung rata-rata per folder
    return np.mean(L_values), np.mean(A_values), np.mean(B_values)


if __name__ == "__main__":
    base_dir = "dataset/raw"
    categories = ["Dark", "Light", "Medium"]

    print("📊 Hasil Analisis Rata-rata CIELAB per Kategori\n")
    print(f"{'Kategori':<10} | {'L* mean':>8} | {'a* mean':>8} | {'b* mean':>8}")
    print("-" * 45)

    results = {}
    for cat in categories:
        folder_path = os.path.join(base_dir, cat)
        result = analyze_folder(folder_path)
        if result:
            L, A, B = result
            results[cat] = (L, A, B)
            print(f"{cat:<10} | {L:8.2f} | {A:8.2f} | {B:8.2f}")

    # hitung ambang batas otomatis antar level
    if all(k in results for k in ["Dark", "Light", "Medium"]):
        L_light, _, _ = results["Dark"]
        L_medium, _, _ = results["Light"]
        L_dark, _, _ = results["Medium"]

        threshold_light_medium = (L_light + L_medium) / 2
        threshold_medium_dark = (L_medium + L_dark) / 2

        print("\n📏 Ambang batas yang disarankan:")
        print(f"Light ↔ Medium : {threshold_light_medium:.2f}")
        print(f"Medium ↔ Dark  : {threshold_medium_dark:.2f}")
