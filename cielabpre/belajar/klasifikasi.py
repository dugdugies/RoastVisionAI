import cv2
import numpy as np
import pandas as pd
import os

CSV_PATH = "dataset/Coffee Bean.csv"         # ubah jika perlu
BASE_DIR = "dataset"                       


def analyze_from_csv(csv_path):
    # baca csv
    df = pd.read_csv(csv_path)

    # kelompokkan berdasarkan labels
    grouped = df.groupby("labels")

    results = {}

    for label, group in grouped:
        L_vals, A_vals, B_vals = [], [], []

        print(f"🔍 Analisis kelas: {label}")

        for _, row in group.iterrows():
            img_path = os.path.join(BASE_DIR, row["filepaths"])

            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Gagal baca gambar: {img_path}")
                continue

            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            L_vals.append(np.mean(l) * (100/255))
            A_vals.append(np.mean(a) - 128)
            B_vals.append(np.mean(b) - 128)

        if len(L_vals) == 0:
            print(f"⚠️ Tidak ada gambar valid untuk kelas {label}")
            continue

        results[label] = (
            np.mean(L_vals),
            np.mean(A_vals),
            np.mean(B_vals)
        )

    return results


if __name__ == "__main__":
    results = analyze_from_csv(CSV_PATH)

    print("\n📊 Hasil Analisis Rata-rata CIELAB")
    print(f"{'Label':<10} | {'L* mean':>8} | {'a* mean':>8} | {'b* mean':>8}")
    print("-" * 45)

    for label, (L, A, B) in results.items():
        print(f"{label:<10} | {L:8.2f} | {A:8.2f} | {B:8.2f}")
