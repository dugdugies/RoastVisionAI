import cv2
import numpy as np
import os
from glob import glob
import pandas as pd

# ======================================================
# 🔧 Fungsi segmentasi: pisahkan biji kopi dari background
# ======================================================
def segment_biji(img):
    # Konversi ke LAB
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(img_lab)

    # Mask berdasarkan L (gelap) dan B (kekuningan)
    mask_L = cv2.inRange(L, 0, 170)
    mask_B = cv2.inRange(B, 130, 190)
    mask = cv2.bitwise_and(mask_L, mask_B)

    # Bersihkan noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Segmentasi hasil
    segmented = cv2.bitwise_and(img, img, mask=mask)
    return segmented, mask


# ======================================================
# 📊 Ekstraksi fitur LAB dari area biji kopi
# ======================================================
def extract_lab_features(img, mask):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(img_lab)

    # Hanya area mask
    L_mean = np.mean(L[mask > 0]) * (100 / 255)
    A_mean = np.mean(A[mask > 0]) - 128
    B_mean = np.mean(B[mask > 0]) - 128

    return L_mean, A_mean, B_mean


# ======================================================
# 💾 Simpan hasil segmentasi dan fitur
# ======================================================
def save_results(segmented_img, mask, output_dir, filename, L, A, B, category):
    os.makedirs(output_dir, exist_ok=True)

    # Simpan gambar hasil segmentasi & mask
    seg_path = os.path.join(output_dir, f"seg_{filename}")
    mask_path = os.path.join(output_dir, f"mask_{filename}")
    cv2.imwrite(seg_path, segmented_img)
    cv2.imwrite(mask_path, mask)

    # Simpan fitur ke CSV
    csv_path = os.path.join("output", "fitur_roasting.csv")
    os.makedirs("output", exist_ok=True)

    # Jika file belum ada → buat header
    header_needed = not os.path.exists(csv_path)

    data = pd.DataFrame([{
        "kategori": category,
        "nama_file": filename,
        "L_mean": round(L, 2),
        "A_mean": round(A, 2),
        "B_mean": round(B, 2)
    }])

    data.to_csv(csv_path, mode='a', index=False, header=header_needed)


# ======================================================
# 🚀 Analisis per folder (kategori roasting)
# ======================================================
def analyze_roast_folder(folder_path, category_name):
    image_paths = glob(os.path.join(folder_path, "*.png")) + glob(os.path.join(folder_path, "*.jpg"))
    if not image_paths:
        print(f"⚠️ Tidak ada gambar di folder {folder_path}")
        return

    print(f"\n============================")
    print(f"Kategori: {category_name}")
    print(f"============================")

    L_vals, A_vals, B_vals = [], [], []

    for path in image_paths:
        img = cv2.imread(path)
        segmented, mask = segment_biji(img)
        L, A, B = extract_lab_features(img, mask)

        filename = os.path.basename(path)
        output_dir = os.path.join("output", "segmentasi", category_name)
        save_results(segmented, mask, output_dir, filename, L, A, B, category_name)

        L_vals.append(L)
        A_vals.append(A)
        B_vals.append(B)

        print(f"📷 {filename} → L*: {L:.2f}, a*: {A:.2f}, b*: {B:.2f}")

    print(f"\n📊 Rata-rata fitur kategori {category_name}:")
    print(f"L*: {np.mean(L_vals):.2f} | a*: {np.mean(A_vals):.2f} | b*: {np.mean(B_vals):.2f}")


# ======================================================
# 🧠 Eksekusi utama
# ======================================================
if __name__ == "__main__":
    base_dir = "dataset/raw"
    categories = ["Dark", "Medium", "Light"]

    for cat in categories:
        folder_path = os.path.join(base_dir, cat)
        analyze_roast_folder(folder_path, cat)

    print("\n✅ Semua hasil tersimpan di folder 'output/' dan file 'fitur_roasting.csv'")
