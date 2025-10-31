import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("dataset/raw/Dark/dark (1).png")
lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
L, a, b = cv2.split(lab)

# Ambil channel b* dan ubah jadi 3-channel biar bisa ditampilkan
b_norm = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
b_rgb = cv2.merge([b_norm, b_norm, b_norm])

plt.imshow(cv2.cvtColor(b_rgb, cv2.COLOR_BGR2RGB))
plt.title("CIELAB b* channel (Blue-Yellow)")
plt.show()
