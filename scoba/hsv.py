import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = "dataset/train/Dark/dark (1).png"
img_path2 = "dataset/train/Light/light (1).png"
img_bgr = cv2.imread(img_path) #konversi bgr
img_bgr2 = cv2.imread(img_path2) #konversi bgr

if img_bgr is None:
    raise ValueError("periksa path Path IMAGE")

#konvert rgb
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_rgb2 = cv2.cvtColor(img_bgr2, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
img_hsv2 = cv2.cvtColor(img_rgb2, cv2.COLOR_RGB2HSV)

#tentukan rentang warna 
lower_hsv_dark = np.array([0, 0, 0])
upper_hsv_dark = np.array([37, 255, 255])
lower_hsv_light = np.array([0, 118, 0])
upper_hsv_light = np.array([179, 255, 255])

# mask 
mask = cv2.inRange(img_hsv, lower_hsv_dark, upper_hsv_dark)
mask2 = cv2.inRange(img_hsv2, lower_hsv_light, upper_hsv_light)

# morph
kernel = np.ones((5,5), np.uint8)
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
mask_clean2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
mask_clean2 = cv2.morphologyEx(mask_clean2, cv2.MORPH_OPEN, kernel)

# result
result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_clean)
result2 = cv2.bitwise_and(img_rgb2,img_rgb2,mask=mask_clean2)

H, S, V = cv2.split(img_hsv)
# hitung statistik warna
H_masked = H[mask_clean > 0]
S_masked = S[mask_clean > 0]
V_masked = V[mask_clean > 0]
# Statistik warna
mean_H, std_H = np.mean(H_masked), np.std(H_masked)
mean_S, std_S = np.mean(S_masked), np.std(S_masked)
mean_V, std_V = np.mean(V_masked), np.std(V_masked)
print("=== Statistik Warna Area Buah (HSV) ===")
print(f"Hue : mean = {mean_H:.2f}, std = {std_H:.2f}")
print(f"Saturation: mean = {mean_S:.2f}, std = {std_S:.2f}")
print(f"Value : mean = {mean_V:.2f}, std = {std_V:.2f}")
# === 7. TAMPILKAN HASIL ===
plt.figure(figsize=(12,8))
plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.title("Gambar Asli (RGB)")
plt.axis("off")
plt.subplot(2,3,2)
plt.imshow(img_hsv)
plt.title("Gambar HSV")
plt.axis("off")
plt.subplot(2,3,3)
plt.imshow(mask, cmap='gray')
plt.title("Mask Awal")
plt.axis("off")
plt.subplot(2,3,4)
plt.imshow(mask_clean, cmap='gray')
plt.title("Mask Setelah Morfologi")
plt.axis("off")
plt.subplot(2,3,5)
plt.imshow(result)
plt.title("Hasil Segmentasi (Area Buah)")
plt.axis("off")
plt.subplot(2,3,6)
plt.imshow(result2)
plt.title("Hasil Segmentasi (Area Buah2)")
plt.axis("off")
plt.tight_layout()
plt.show()