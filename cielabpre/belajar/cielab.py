import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("dataset/raw/Light/light (8).png")# fungsi baca gambar dari cv2
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #konversi gambar jadi rgb

img_lab = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2LAB)

l, a, b = cv2.split(img_lab)
Lmean = np.mean(l)
Amean = np.mean(a)
Bmean = np.mean(b)
L_mean_real = Lmean * (100 / 255)
A_mean_real = Amean - 128
B_mean_real = Bmean - 128

print(f"L*: {L_mean_real:.2f}")
print(f"a*: {A_mean_real:.2f}")
print(f"b*: {B_mean_real:.2f}")

# fig, ax = plt.subplots(1, 4, figsize=(12, 4))
# ax[0].imshow(img_rgb)
# ax[0].set_title("RGB")

# ax[1].imshow(l, cmap="gray")
# ax[1].set_title("L* (Lightness)")

# ax[2].imshow(a, cmap="gray")
# ax[2].set_title("a* (Green-Red)")

# ax[3].imshow(b, cmap="gray")
# ax[3].set_title("b* (Blue-Yellow)")

# for a in ax: a.axis("off")
# plt.show()