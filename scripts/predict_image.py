import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

model = load_model('models/CoffeeRoast_MobileNetV2_YCbCr_final.keras')

img_path = 'dataset/sample/dark/dark_sample.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = cv2.cvtColor(img_array.astype('uint8'), cv2.COLOR_RGB2YCrCb)
img_array = np.expand_dims(img_array, axis=0) / 255.0

with open('models/class_indices.json') as f:
    import json
    class_indices = json.load(f)

# pastikan urutan sesuai
classes = list(class_indices.keys())
print("Urutan kelas:", classes)

pred = model.predict(img_array)
print("Prediksi:", classes[np.argmax(pred)])
print("Confidence:", round(np.max(pred) * 100, 2), "%")
