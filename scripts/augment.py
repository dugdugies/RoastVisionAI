import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np

# =============================================================
# KONFIGURASI
# =============================================================
input_folder = "dataset primer/"                # folder dataset asli
output_folder = "augmented_dataset"     # folder untuk menyimpan hasil augmentasi
target_per_class = 300                  # Ubah sesuai target kamu


datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.10,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)

# =============================================================
# FUNGSI AUGMENTASI
# =============================================================
def augment_class(class_name):
    class_input_path = os.path.join(input_folder, class_name)
    class_output_path = os.path.join(output_folder, class_name)
    
    os.makedirs(class_output_path, exist_ok=True)

    images = os.listdir(class_input_path)
    current_count = len(images)
    print(f"Class {class_name}: {current_count} images found.")

    if current_count >= target_per_class:
        print(f" -> Class {class_name} sudah memenuhi target ({target_per_class}), tidak perlu augmentasi.\n")
        return

    needed = target_per_class - current_count
    print(f" -> Perlu menambah {needed} gambar.\n")

    i = 0
    while i < needed:
        img_name = images[i % current_count]  # looping dataset asli
        img_path = os.path.join(class_input_path, img_name)

        img = load_img(img_path)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # generate 1 gambar per iterasi
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=class_output_path,
                                  save_prefix=class_name,
                                  save_format='jpg'):
            i += 1
            break  # generate tepat 1 gambar

    print(f" -> DONE Class {class_name}: Total jadi {target_per_class} gambar.\n")


# =============================================================
# JALANKAN UNTUK SEMUA KELAS
# =============================================================

classes = ["Dark", "Green", "Light", "Medium"]

for cls in classes:
    augment_class(cls)

print("=== SELESAI AUGMENTASI ===")