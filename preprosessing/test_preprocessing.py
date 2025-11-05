import cv2
import os
from tqdm import tqdm

# --- PATH INPUT & OUTPUT ---
input_base = "dataset/raw"  # ubah ke path dataset kamu
output_base = "dataset/preprocessed"

classes = ["Light", "Medium", "Dark"]


for cls in classes:
    input_folder = os.path.join(input_base, cls)
    output_folder = os.path.join(output_base, cls)
    os.makedirs(output_folder, exist_ok=True)
    
    files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
    print(f"🔄 Memproses kelas: {cls} ({len(files)} gambar)")

    for filename in tqdm(files):
        img_path = os.path.join(input_folder, filename)
        save_path = os.path.join(output_folder, filename)

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Gagal membaca {img_path}, dilewati.")
            continue
        
        # resize biar seragam
        img = cv2.resize(img, (224, 224))
        
        # konversi ke YCbCr
        ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        
        # simpan hasil
        cv2.imwrite(save_path, ycbcr)

print("\n✅ Semua gambar berhasil diproses dan disimpan ke:", output_base)
