from keras.models import load_model
from keras.preprocessing import image
import numpy as np

model = load_model('models/coffee_phase1_best.keras')

img_path = 'dataset/sample/medium/medi.png' 
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

pred = model.predict(img_array)
classes = ['dark','medium','light']

print("Prediksi:", classes[np.argmax(pred)])
print("Confidence:", round(np.max(pred) * 100, 2), "%")
