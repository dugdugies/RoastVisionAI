import cv2
import matplotlib.pyplot as plt
import os

# --- PATH DATASET ---
base_dir = "dataset/raw"  # ubah sesuai lokasi dataset kamu
classes = ["Light", "Medium", "Dark"]

# ambil 1 gambar per kelas untuk contoh
sample_paths = {}
for cls in classes:
    folder = os.path.join(base_dir, cls)
    files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
    if not files:
        raise FileNotFoundError(f"Tidak ada gambar di folder {folder}")
    sample_paths[cls] = os.path.join(folder, files[0])  # ambil 1 gambar pertama

# --- TAMPILKAN PREPROCESSING ---
plt.figure(figsize=(15, 10))
i = 1

for cls, path in sample_paths.items():
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Gagal membaca gambar {path}")
    img = cv2.resize(img, (224, 224))
    
    # Konversi ke berbagai ruang warna
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # BGR → RGB buat tampil di matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # tampilkan grid: tiap baris = satu kategori
    plt.subplot(len(classes), 4, i); plt.imshow(img_rgb); plt.axis("off"); plt.title(f"{cls} - Original")
    plt.subplot(len(classes), 4, i+1); plt.imshow(lab); plt.axis("off"); plt.title(f"{cls} - LAB")
    plt.subplot(len(classes), 4, i+2); plt.imshow(ycbcr); plt.axis("off"); plt.title(f"{cls} - YCbCr")
    plt.subplot(len(classes), 4, i+3); plt.imshow(hsv); plt.axis("off"); plt.title(f"{cls} - HSV")
    i += 4

plt.tight_layout()
plt.show()
