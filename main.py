from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from flask_mqtt import Mqtt
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import logging
from database import init_db, insert_result, get_today_results, get_counts_today
import time

UPLOAD_FOLDER = 'dataset/esp32-cam'
FIXED_FILENAME = 'latest.jpg'
MODEL_PATHS = [
    "models/model_dagi/cobamodel/cnn_final_model.keras"
]
IMG_SIZE = (128, 128)
ALLOWED_EXT = {'jpg', 'jpeg', 'png'}

MQTT_TOPIC_RESULT = "hasil/roasting"

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MQTT_BROKER_URL'] = 'test.mosquitto.org'
app.config['MQTT_BROKER_PORT'] = 1883
app.config['MQTT_REFRESH_TIME'] = 1.0

mqtt = Mqtt(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RoastVisionAI")

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logger.info(f"Upload folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")

init_db()

# Load model
model = None
for p in MODEL_PATHS:
    if os.path.exists(p):
        try:
            logger.info(f"Attempting to load model from: {p}")
            model = tf.keras.models.load_model(p)
            logger.info(f"✅ Loaded model: {p}")
            break
        except Exception as e:
            logger.exception(f"Failed to load model from {p}: {e}")
    else:
        logger.warning(f"Model path not found: {p}")

if model is None:
    logger.error("No model could be loaded. Exiting.")
    raise SystemExit("Model not found or failed to load. Check MODEL_PATHS and model files.")

LABELS = ["Green", "Light", "Medium", "Dark"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def predict_image_raw(img_pil):
    # Hanya convert ke RGB dan resize
    img_rgb = img_pil.convert('RGB')
    img_resized = img_rgb.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    
    # Convert to array dan normalize (0-1)
    arr = np.array(img_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Predict
    preds = model.predict(arr, verbose=0)
    probs = preds[0] if preds.ndim == 2 else preds
    idx = int(np.argmax(probs))
    confidence = float(np.max(probs))
    label = LABELS[idx] if idx < len(LABELS) else str(idx)
    
    return label, confidence, probs.tolist()

@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("Connected to MQTT broker")
        mqtt.subscribe("roasting/kopi")
    else:
        logger.error("Failed to connect to MQTT broker, rc=%s", rc)

@mqtt.on_message()
def handle_message(client, userdata, message):
    try:
        payload = message.payload.decode()
    except Exception:
        payload = str(message.payload)
    logger.info("MQTT message received -> %s : %s", message.topic, payload)

    if message.topic == MQTT_TOPIC_RESULT:
        try:
            data = json.loads(payload)
            filename = data.get('file')
            prediction = data.get('prediction')
            confidence = int(data.get("confidence", 0))
            if filename and prediction:
                insert_result(filename, prediction, confidence)
                logger.info("Inserted MQTT result into DB: %s", filename)
        except Exception:
            logger.exception("Failed to parse/persist MQTT payload")

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.post("/foto")
def index():
    try:
        # Cek apakah ada data gambar
        if not request.data:
            return jsonify({"error": "no image received (raw bytes empty)"}), 400

        logger.info(f"Received image: {len(request.data)} bytes")

        # Baca gambar langsung dari bytes
        try:
            img_pil = Image.open(io.BytesIO(request.data))
            logger.info(f"Image size: {img_pil.size}, mode: {img_pil.mode}")
        except Exception as e:
            logger.exception("Failed to open image: %s", e)
            return jsonify({"error": "invalid image data", "detail": str(e)}), 400

        # Prediksi TANPA preprocessing
        try:
            label, confidence, probs = predict_image_raw(img_pil)
            confidence_percent = int(confidence * 100)
            
            # Format probabilitas semua kelas
            probs_formatted = {LABELS[i]: round(probs[i]*100, 1) for i in range(len(LABELS))}
            probs_string = ", ".join([f"{k}: {v}%" for k, v in probs_formatted.items()])
            
            logger.info("="*70)
            logger.info(f"PREDICTION: {label} ({confidence_percent}%)")
            logger.info(f"All probabilities: {probs_string}")
            logger.info("="*70)
            
        except Exception as e:
            logger.exception("❌ Prediction failed: %s", e)
            return jsonify({"error": "prediction failed", "detail": str(e)}), 500

        # Simpan gambar (overwrite setiap kali)
        filename = FIXED_FILENAME
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            img_pil.convert("RGB").save(save_path, format="JPEG", quality=95)
            logger.info(f"Image saved: {save_path}")
        except Exception as e:
            logger.exception("Failed to save image: %s", e)
            # Tetap lanjut meskipun gagal save

        # Save ke database
        try:
            insert_result(filename, label, confidence_percent)
            logger.info("Result saved to database")
        except Exception:
            logger.exception("DB insert error")

        # Publish ke MQTT
        payload = {
            "file": filename,
            "prediction": label,
            "confidence": confidence_percent,
            "all_probabilities": probs_formatted
        }
        try:
            mqtt.publish(MQTT_TOPIC_RESULT, json.dumps(payload))
            logger.info("Published to MQTT")
        except Exception as e:
            logger.exception("Failed to publish MQTT: %s", e)

        # Response
        return jsonify({
            "status": "ok",
            "file": filename,
            "prediction": label,
            "confidence": confidence_percent,
            "probabilities": probs_formatted
        }), 200

    except Exception as e:
        logger.exception("Unexpected error in /foto: %s", e)
        return jsonify({"error": "internal server error", "detail": str(e)}), 500

@app.route('/api/results/today')
def api_result_today():
    """Get hasil prediksi hari ini"""
    rows = get_today_results()
    items = []
    for filename, pred, conf, created_at in rows:
        items.append({
            'file_url': f"/uploads/{filename}",
            'filename': filename,
            'prediction': pred,
            'confidence': conf,
            'created_at': created_at
        })
    counts = get_counts_today()
    return jsonify({'results': items, 'counts': counts})

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/logo')
def logo():
    return send_from_directory('templates', 'logo-kopi.png')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=8000)